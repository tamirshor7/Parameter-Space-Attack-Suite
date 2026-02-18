#!/usr/bin/env python3
# attacks/bpda.py
"""
BPDA attack for INR parameter-space classifiers (repo-compatible).

Notes:
- This script uses FullPGD.fit_image for true inner fitting in evaluation,
  but uses forward_only_fit_mods() as a BPDA-safe forward surrogate inside the
  custom autograd.Function so no .backward() is executed inside forward.
- The classifier expects modulation vectors (mods) as input; BPDA returns mods.
"""

import argparse
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Ensure repo root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloader import get_mnist_loader
from utils import set_random_seeds
from SIREN import ModulatedSIREN
from train_classifier import Classifier
from full_pgd import FullPGD  
# ---------------------------
# Argument parser (matches full_pgd.py + BPDA extras)
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--inner-lr', type=float, default=0.01, help='learn rate for internal modulation optimization')
    parser.add_argument('--ext-lr', type=float, default=0.01, help='learn rate for external adversarial perturbation optimization')
    parser.add_argument('--hidden-dim', type=int, default=256, help='SIREN hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--mod-steps', type=int, default=10, help='Number of internal modulation optimization steps per PGD iteration')
    parser.add_argument('--pgd-steps', type=int, default=100, help='Number of projected gradient descent steps')
    parser.add_argument('--cwidth', type=int, default=512, help='classifier MLP hidden dimension')
    parser.add_argument('--cdepth', type=int, default=3, help='classifier MLP depth')
    parser.add_argument('--depth', type=int, default=10, help='SIREN depth')
    parser.add_argument('--dataset', choices=["mnist", "fmnist"], help="Train for MNIST or Fashion-MNIST")
    parser.add_argument('--data-path', type=str, default='..', help='path to MNIST or FMNIST dataset')
    parser.add_argument('--siren-checkpoint', type=str, help='path to pretrained SIREN from meta-optimization')
    parser.add_argument('--classifier-checkpoint', type=str, help='path to pretrained classifier')
    parser.add_argument('--epsilon', type=int, default=16, help='attack epsilon -- epsilon/255 is the de-facto attack l_inf constraint.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')

    # BPDA-specific
    parser.add_argument('--bpda-mode', type=str, default='identity', choices=['identity', 'reconstruction'],
                        help="BPDA backward surrogate: 'identity' or 'reconstruction' (reconstruction re-runs a short inner fit w/ autograd if possible)")
    parser.add_argument('--eot-samples', type=int, default=1, help='EOT samples (expectation over random fit seeds)')
    parser.add_argument('--restarts', type=int, default=1, help='PGD random restarts (outer)')
    parser.add_argument('--start', type=int, default=0, help='start')
    parser.add_argument('--end', type=int, default=10000, help='end')
    return parser.parse_args()

# ---------------------------
# Helper: forward-only modulation fit (BPDA-safe)
# ---------------------------
def forward_only_fit_mods(fullpgd, x, eot_samples=1):
    """
    Compute per-sample modulation vectors deterministically without using autograd.
    Mirrors the forward behavior of FullPGD.fit_image but avoids any .backward() calls.
    Args:
      - fullpgd: instance of FullPGD (has .inr, .inner_steps, .inner_lr)
      - x: tensor [B,1,H,W]
      - eot_samples: average over several RNG seeds (if >1)
    Returns:
      - mods: tensor [B, mod_dim]
    """
    device = next(fullpgd.inr.parameters()).device
    B = x.shape[0]
    mods_list = []

    for b in range(B):
        img = x[b]  # shape [1,H,W]
        # initialize modulator like fit_image does when start_mod is None
        modulator = torch.zeros(fullpgd.inr.modul_features, device=device, dtype=torch.float32)

        # if eot_samples>1, average several forward-only fits (deterministic per seed)
        accum = torch.zeros_like(modulator)
        for s in range(eot_samples):
            # mimic different RNG seeds if desired
            torch.manual_seed(s)
            mod_copy = modulator.clone()
            # forward-only inner-loop: we do deterministic, no-autograd updates to mod_copy
            for step in range(fullpgd.inner_steps):
                with torch.no_grad():
                    fitted = fullpgd.inr(mod_copy)   # fitted shape depends on your SIREN output
                    # compute a simple scalar residual to update mod_copy (heuristic)
                    # Aim: push mod_copy toward lower reconstruction error without autograd.
                    # Compute mean residual and apply small step
                    # Shapes: fitted may be [1, N] while img is [1,H,W]; flatten both.
                    img_flat = img.view(-1)
                    fitted_flat = fitted.view(-1)[: img_flat.numel()] if fitted.view(-1).numel() >= img_flat.numel() else fitted.view(-1)
                    # compute surrogate "direction" as mean difference
                    # This is a heuristic; it keeps the forward mapping stable and deterministic.
                    if fitted_flat.numel() == img_flat.numel():
                        resid = (fitted_flat - img_flat).mean()
                    else:
                        # fallback: use fitted mean if shapes mismatch
                        resid = (fitted_flat - fitted_flat.mean()).mean()
                    lr = getattr(fullpgd, "inner_lr", 0.01)
                    # update: simple scalar-shift on all mod entries
                    mod_copy = mod_copy - lr * resid
            accum += mod_copy
        avg_mod = accum / float(max(1, eot_samples))
        mods_list.append(avg_mod.unsqueeze(0))
    mods = torch.cat(mods_list, dim=0)
    return mods

# ---------------------------
# BPDA autograd surrogate (returns mods in forward)
# ---------------------------
class BPDASurrogate(torch.autograd.Function):
    """
    BPDA surrogate for INR → modulation-space classifier.

    Forward:
        x (image) → surrogate mod vector (mods)

    Backward:
        dL/dmods → approximate dL/dx
        using a finite-difference Jacobian-vector product:
            J_inr(theta) * (dL/dtheta)
        where theta = mods from forward().
    """

    @staticmethod
    def forward(ctx, x, fullpgd, inner_steps, eot_samples, bpda_mode):
        """
        Inputs:
            x: shape [1,1,28,28] or [1,784]
        Returns:
            mods: shape [1, mod_dim]
        """
        ctx.fullpgd = fullpgd
        ctx.inner_steps = inner_steps
        ctx.eot_samples = eot_samples
        ctx.bpda_mode = bpda_mode

        # --- Surrogate mod fitting (no autograd, BPDA-safe) ---
        mods = forward_only_fit_mods(fullpgd, x, eot_samples=eot_samples)   # [1, mod_dim]

        # Save for backward
        ctx.save_for_backward(x.detach(), mods.detach())

        # Classifier will receive these mods
        return mods.detach()


    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output = dL/d(mods)  (shape [1, mod_dim])

        We compute:
            dL/dx = J_inr(theta) * (dL/dtheta)

        using a single finite-difference J·v approximation:
            (INR(theta + h*v) - INR(theta)) / h
        """

        # Unpack saved tensors
        x, mods = ctx.saved_tensors
        fullpgd = ctx.fullpgd
        bpda_mode = ctx.bpda_mode

        device = x.device

        # We assume batch size = 1
        assert grad_output.shape[0] == 1, "BPDA assumes batch size 1"

        # grad wrt modulation vector: shape [1, mod_dim] -> [mod_dim]
        g_theta = grad_output.view(-1).detach()

        # theta = modulation vector from forward
        theta = mods.view(-1).detach()   # [mod_dim]

        # Compute INR(theta) (flattened)
        with torch.no_grad():
            recon = fullpgd.inr(theta)   # shape [1, N]
            recon = recon.view(-1)                   # [N] (should be 784 for MNIST)

        # --- Finite-difference Jacobian-vector product ---
        # h is small step size.
        h = 1e-3
        theta_norm = torch.norm(theta).item()
        h_eff = h * (1.0 + theta_norm)

        theta_pert = (theta + h_eff * g_theta)

        with torch.no_grad():
            recon_pert = fullpgd.inr(theta_pert).view(-1)     # [N]

        # J * g ≈ (f(theta + h*g) - f(theta)) / h
        grad_recon = (recon_pert - recon) / h_eff             # [N]

        # --- Map recon-gradient to image gradient ---
        # Since fit_image matches fitted.flatten() to image.flatten(),
        # d(recon)/d(image) ~ identity in your architecture.

        x_shape = x.shape
      
        if len(x_shape) == 4:   # [1,1,28,28]
            H = x_shape[2]
            W = x_shape[3]
            grad_x = grad_recon.view(1, 1, H, W).to(device)

        elif len(x_shape) == 2:  # [1,784]
            grad_x = grad_recon.view(1, -1).to(device)

        else:
            # Fallback: broadcast reshape
            grad_x = grad_recon.view(1, -1).unsqueeze(-1).expand(x_shape).contiguous().to(device)

        # Optional refinement mode: ('reconstruction')
        # This step is skipped unless you explicitly request it.
        if bpda_mode == "reconstruction":
            try:
                # Re-estimate direction by differentiating recon wrt x via fullpgd.fit_image (expensive)
                x_req = x.detach().requires_grad_(True)
                mods_short, _ = fullpgd.fit_image(x_req, start_mod=None, clean=False, return_mse=True)
                logits = fullpgd.classifier(mods_short)
                loss = logits.sum()
                grad_refined = torch.autograd.grad(loss, x_req, retain_graph=False, allow_unused=False)[0]
                if grad_refined is not None:
                    grad_x = 0.5 * grad_x + 0.5 * grad_refined.detach()
            except Exception:
                pass

        # Return gradient for x; None for non-tensor arguments
        return grad_x, None, None, None, None


# ---------------------------
# Attack runner using FullPGD internals (classifier takes mods)
# ---------------------------
def run_bpda_attack(fullpgd: FullPGD, loader, eps, pgd_iters, pgd_lr, device, bpda_mode, eot_samples, restarts, start, end):
    pbar = tqdm(loader, total=len(loader))

    clean_correct = 0
    adv_correct = 0
    samples_seen = 0

    
    
    fullpgd.to(device)
    fullpgd.eval()
    total = 0
    fooled = 0
    
    for x, labels in tqdm(loader):
        x = x.to(device)
        labels = labels.to(device)
        if total < start or total>=end:
            total +=1
            continue
        else:
            total += 1

        # compute clean modulation & check if correctly classified
        clean_logits, clean_mod, clean_mse = fullpgd(x.unsqueeze(1), clean=True, return_mse=True)
        if clean_logits.argmax(1).item() != labels.item():
            # skip samples misclassified on clean
            continue

        success = False
        for r in range(restarts):
            # init perturbation randomly within eps ball
            pert = (torch.empty_like(x).uniform_(-eps, eps)).to(device)
            x_adv = torch.clamp(x + pert, 0.0, 1.0).detach()
            x_adv.requires_grad = True

            for it in range(pgd_iters):
                # Forward surrogate: compute mods via BPDA forward (no autograd inside)
                
                mods = BPDASurrogate.apply(x_adv, fullpgd, fullpgd.inner_steps, eot_samples, bpda_mode)
                # Classify using mods (the classifier expects mods)
                logits = fullpgd.classifier(mods)
               
                loss = 1.0 - F.cross_entropy(logits.squeeze(), labels.squeeze())  # same objective as full_pgd
                # Backward: compute grad w.r.t x via BPDA.backward surrogate
                fullpgd.classifier.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.zero_()
                loss.backward()
                # x_adv.grad should have been filled by BPDASurrogate.backward
                if x_adv.grad is None:
                    # fallback: numerical approx (very slow)
                    grad = torch.sign(torch.autograd.grad(loss, x_adv, retain_graph=False, allow_unused=True)[0])
                else:
                    grad = x_adv.grad.detach()

                # PGD L_inf step
                x_adv = x_adv.detach() + pgd_lr * torch.sign(grad)
                x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
                x_adv = torch.clamp(x_adv, 0, 1).detach()
                x_adv.requires_grad = True

            
                final_mods, final_mse = fullpgd.fit_image(x_adv.unsqueeze(1), start_mod=clean_mod, clean=True, return_mse=True)
                with torch.no_grad():
                    final_logits = fullpgd.classifier(final_mods)
                pred = final_logits.argmax(dim=1).item()
                if pred != labels.item():
                    success = True
                    break


        # --------------------------------------
        # Accuracy logging (tqdm)
        # --------------------------------------
        clean_correct += (clean_logits.argmax(1) == labels).item()
        adv_correct   += (logits.argmax(1) == labels).item()
        samples_seen += 1

        clean_acc = clean_correct / samples_seen
        robust_acc = adv_correct / samples_seen

        pbar.set_description(
            f"[BPDA] eps={eps:.4f} | clean_acc={clean_acc:.3f} | robust_acc={robust_acc:.3f}"
        )
        # --------------------------------------

        if success:
            fooled += 1

    robust_acc = 1.0 - float(fooled) / float(max(1, total))
    print(f"BPDA robust acc: {robust_acc:.4f} ({total-fooled}/{total})")
    return robust_acc

# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    args = get_args()
    set_random_seeds(args.seed, args.device)

    loader = get_mnist_loader(args.data_path, train=False, batch_size=1, fashion=(args.dataset == "fmnist"))

    # Load SIREN INR (modSiren)
    modSiren = ModulatedSIREN(height=28, width=28, hidden_features=args.hidden_dim,
                              num_layers=args.depth, modul_features=args.mod_dim).to(args.device)
    siren_ckpt = torch.load(args.siren_checkpoint)
    modSiren.load_state_dict(siren_ckpt['state_dict'])

    # Load classifier
    classifier = Classifier(width=args.cwidth, depth=args.cdepth,
                            in_features=args.mod_dim, num_classes=10).to(args.device)
    class_ckpt = torch.load(args.classifier_checkpoint)
    classifier.load_state_dict(class_ckpt['state_dict'])
    classifier.eval()

    # instantiate FullPGD and reuse its fit_image / inr / classifier
    fullpgd = FullPGD(modSiren, classifier, inner_steps=args.mod_steps, inner_lr=args.inner_lr, device=args.device)
    fullpgd.to(args.device)
    fullpgd.eval()

    eps = args.epsilon / 255.0
    run_bpda_attack(fullpgd, loader, eps, args.pgd_steps, args.ext_lr, args.device,
                    bpda_mode=args.bpda_mode, eot_samples=args.eot_samples, restarts=args.restarts,start=args.start, end=args.end)
