import os
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import get_mnist_functa, get_mnist_loader
from dataloader_modelnet import get_modelnet_loader
from utils import adjust_learning_rate, set_random_seeds, BinaryMask
from tqdm import tqdm
import numpy as np
import argparse
from SIREN import ModulatedSIREN, ModulatedSIREN3D
from train_classifier import Classifier
from matplotlib import pyplot as plt
from higher import get_diff_optim


class ICOP(nn.Module):
    def __init__(self, inr, classifier, inner_steps = 100, inner_lr = 0.01, voxels = False, device='cuda'):
        """
        :param inr: pretrained inr model
        :param classifier: 'clean' pretrained classifier model to attack
        :param inner_steps: number of modulation optimization steps in full (non-truncated) optimization.
        :param inner_lr: learn rate for internal modulation optimization.
        :param: voxels: use 3d data
        :param device: use cuda or cpu.
        """
        super(ICOP, self).__init__()
        #load classifier
        self.classifier = classifier.to(device)
        self.classifier.eval()
        
        #load inr
        self.inr = inr.to(device)
        self.inr.eval()
        
        #optimization params
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.inner_optimizer_fn = optim.Adam if voxels else optim.SGD
        
        self.voxels = voxels
        
       
    
    def fit_image(self, x, return_mse = False):
        '''optimize and return modulation for a specific (potentially perturbed) signal-domain input x'''
        device = "cuda"
        inner_criterion = nn.MSELoss().cuda()
        
        
        mses = []
        mods = None
        # Inner Optimization.
        for image in x:
            modulator = torch.zeros(self.inr.modul_features).float().to(device)
         
            modulator.requires_grad = True
            if not self.voxels:
                image = image[0]
            image = image.view(1, -1).T.to(device)
            
            inner_optimizer = self.inner_optimizer_fn([modulator],lr=self.inner_lr)
           
            mse = 0
            for step in range(self.inner_steps):
             
                inner_optimizer.zero_grad()
                fitted = self.inr(modulator)
                
                inner_loss = inner_criterion(fitted.T, image.flatten()[None]) if self.voxels else inner_criterion(fitted.flatten(), image.flatten())
                mse = inner_loss.item()
                inner_loss.backward()
                # Clip the gradient.
                torch.nn.utils.clip_grad_norm_([modulator], 1)
                # Update.
                inner_optimizer.step()
                    
        
            mses.append(mse)
            mods = modulator.unsqueeze(0) if mods is None else torch.cat((modulator.unsqueeze(0),mods),axis=0)
      
        
        return torch.flip(mods,dims=[0]) if not return_mse else (torch.flip(mods,dims=[0]), mse)
        
    
    
    def forward(self, x, return_mse = False):
        '''find modulation for x and classify it.
           x is input in signal domain, start_mod is optional starting point for modulation optimization, return_mse indicates whether to return the best modulation representation error.'''
    
        modulations = self.fit_image(x, return_mse)
        if return_mse:
            modulations,mse = modulations
     
        preds = self.classifier(modulations)
        if return_mse:
            return preds, modulations.detach(), mse
        else:
            return preds, modulations.detach()


    
def run_attack(model, loader, criterion, constraint, mod_dim, pgd_iters, pgd_lr, proj_lr, max_proj_iters, voxels = False, device='cuda'):
   
    """
    :param model: attack model holding pretrained INR and classifier.
    :param loader: signal domain data-loader for data for which we optimize attacks.
    :param criterion: clean classifier optimization criterion.
    :param constraint: l_inf attack bound.
    :param: mod_dim: dimension of modulation vectors for SIREN.
    :param pgd_iters: number of PGD iterations.
    :param pgd_lr: learn-rate for PGD attack.
    :param proj_lr: learn-rate for feasible projection.
    :param max_proj_iters: max num of feasible projection iterations.
    :param voxels: use voxel-grid data if True
    :param device: use cuda or cpu.
    """
    prog_bar = tqdm(loader, total=len(loader))
    acc_sum = 0
    samples_num = 0
    rights_clean = 0
    rights_attacked = 0
    samples=0
    projection_loss_eps = constraint/100 #for stop criteria of iterative projection
    
    for x,labels in prog_bar:
    
        samples += 1
        x = x.cuda()
        labels = labels.cuda()
        
        
        clean, clean_mod, clean_mse = model(x.unsqueeze(1), return_mse = True)
        rights_clean += (clean.argmax(1)==labels).item()
        labels = labels.cuda()
        pert = torch.zeros(mod_dim).to(device)
        pert.requires_grad = True
        optimizer = torch.optim.Adam([pert],lr=pgd_lr)
        prev_change_voxels = None
        
        for iter in range(pgd_iters):
            
            optimizer.zero_grad() 
            x_mod = model.fit_image(x).detach().squeeze() #find clean modulation without backpropping through optimization steps
            output = model.classifier(x_mod+pert[None]).squeeze()
            cls_loss = 1-criterion(output[None],labels)
            fitted = model.inr(x_mod+pert).reshape(1,15,15,15) if voxels else model.inr(x_mod+pert) #15X15X15 is resampling dimension for voxels grids

            rec_loss = torch.nn.functional.l1_loss(x,fitted) if voxels else torch.nn.functional.l1_loss(x.flatten().unsqueeze(1),fitted)
            dldr = torch.autograd.grad(rec_loss,pert)[0] 
            dldc = torch.autograd.grad(cls_loss,pert)[0]
            projected_grad = dldc - (dldc@dldr / (dldr@dldr) * dldr)
            with torch.no_grad():
                pert.grad = projected_grad
            optimizer.step()
             
            #project weight-space perturbation onto feasible set in signal domain
            if voxels:
 
                #project onto feasible
                with torch.no_grad():
                    drop_threshold = torch.topk(fitted.flatten(), int(constraint), largest=True).values.min()
                    pd = (fitted >= drop_threshold).to(torch.float)
                    cur_change_voxels = torch.where(pd.flatten()==1)[0]
                    if prev_change_voxels is not None and (~torch.eq(cur_change_voxels ,prev_change_voxels)).sum().item():
                        consecutive_no_change_iters += 1
                    else:
                        consecutive_no_change_iters = 0
                        
                prev_change_voxels = cur_change_voxels[0].clone()
                if consecutive_no_change_iters == max_proj_iters:
                    break
                
            else:

                pert = pert.detach()
                pert.requires_grad = True
                fitted_pert = model.inr(pert)
                proj_loss = torch.nn.ReLU()(fitted_pert.abs().max()-constraint)
                dlinf_dp = torch.autograd.grad(proj_loss,pert)[0]
                
                projection_iters = 0
                while projection_iters < max_proj_iters and fitted_pert.abs().max()-projection_loss_eps>constraint:
                    projection_iters += 1
                    pert = pert - proj_lr*dlinf_dp
                    fitted_pert = model.inr(pert.detach())
                pert = pert.detach()
                pert.requires_grad = True  
            
          
    
        
        
        
        #Final Evaluation for perturbation
        if voxels:
            drop_threshold = torch.topk(fitted.flatten(), int(constraint), largest=True).values.min()
            with torch.no_grad():
                fitted = (fitted >= drop_threshold).to(torch.float)
            
            fitted = x + fitted - 2*fitted*x
            fitted_inr = model.fit_image(fitted.detach()).detach().squeeze()
          
            with torch.no_grad():
                output = model.classifier(fitted_inr[None])[0]
                final_mse = torch.nn.functional.mse_loss(x.flatten()[None],model.inr(fitted_inr).flatten()[None]).item()
        else:
            fitted_pert = model.inr(pert.detach())
            fitted_pert = torch.clamp(fitted_pert,-constraint,constraint)
            output, cur_mod, final_mse = model((x+fitted_pert.reshape(*x.shape)).detach().unsqueeze(1), return_mse = True)
            
 
        print(f"Clean MSE: {clean_mse}. Perturbation Final MSE: {final_mse}")
        with torch.no_grad():
            attack_succ = (output.argmax(1-voxels) != labels).item()
            if not attack_succ:
                rights_attacked += 1
            
            
            
        prog_bar.set_description(f'Constraint {constraint}: Curr clean acc {rights_clean/samples}; Curr attacked acc {rights_attacked/samples}.')
        
    prog_bar.set_description(f'Constraint {constraint}: Final clean acc {rights_clean/samples}; Final attacked acc {rights_attacked/samples}.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--inner-lr', type=float, default=0.01, help='learn rate for internal modulation optimization')
    parser.add_argument('--ext-lr', type=float, default=0.01, help='learn rate for external adversarial perturbation optimization')
    parser.add_argument('--proj-lr', type=float, default=0.0005, help='learn rate for projecting weight-space perturbation to feasible constraints in signal domain.')
    parser.add_argument('--max-proj-iters', type=float, default=10, help='maximum number of iterations for projecting weight-space perturbation to feasible\
                                                                          constraints in signal domain.')
    parser.add_argument('--hidden-dim', type=int, default=256, help='SIREN hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--mod-steps', type=int, default=10, help='Number of internal modulation optimization steps per PGD iteration, without truncation (for clean evaluation)')
    parser.add_argument('--pgd-steps', type=int, default=100, help='Number of projected gradient descent steps')
    parser.add_argument('--cwidth', type=int, default=512, help='classifier MLP hidden dimension')
    parser.add_argument('--cdepth', type=int, default=3, help='classifier MLP depth')
    parser.add_argument('--depth', type=int, default=10, help='SIREN depth')
    parser.add_argument('--dataset', choices=["mnist", "fmnist","modelnet"], help="Train for MNIST, Fashion-MNIST or ModelNet10") 
    parser.add_argument('--data-path', type=str, default='..', help='path to MNIST,FMNIST or ModelNet10 dataset')
    parser.add_argument('--siren-checkpoint', type=str, help='path to pretrained SIREN from meta-optimization')
    parser.add_argument('--classifier-checkpoint', type=str, help='path to pretrained classifier')
    parser.add_argument('--epsilon', type=int, default=16, help='attack epsilon -- epsilon/255 is the de-facto attack l_inf constraint for 2D data. Directly l_0 norm for 3D data.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    return parser.parse_args()
    
if __name__ == '__main__':

    
    args = get_args()    
    set_random_seeds(args.seed, args.device)
    
    #Initiallize pretrained models 
    if args.dataset == "modelnet":
        resample_shape = (15,15,15) #we use this resampling in all experiments
        dataloader = get_modelnet_loader(train=False, batch_size=1, resample_shape=resample_shape)
        modSiren = ModulatedSIREN3D(height=resample_shape[0], width=resample_shape[1], depth=resample_shape[2],\
            hidden_features=args.hidden_dim, num_layers=args.depth, modul_features=args.mod_dim) #we use a mod dim of 2048 in our exps
  
    else:        
        dataloader = get_mnist_loader(args.data_path, train=False, batch_size=1, fashion = args.dataset=="fmnist")
        modSiren = ModulatedSIREN(height=28, width=28, hidden_features=args.hidden_dim, num_layers=args.depth, modul_features=args.mod_dim) #28,28 is mnist and fmnist dims
  
    pretrained = torch.load(args.siren_checkpoint)
    modSiren.load_state_dict(pretrained['state_dict'])
    
    classifier = Classifier(width=args.cwidth, depth=args.cdepth,
                            in_features=args.mod_dim, num_classes=10,batchnorm=args.dataset=="modelnet").to(args.device)
    pretrained = torch.load(args.classifier_checkpoint)
    classifier.load_state_dict(pretrained['state_dict'])
    classifier.eval()
    
    attack_model = ICOP(modSiren, classifier, inner_steps=args.mod_steps, inner_lr = args.inner_lr, voxels = args.dataset=='modelnet',device=args.device)
    attack_model.to(args.device)
    constraint = int(args.epsilon) if args.dataset=="modelnet" else args.epsilon/255 #for the 3D case the constraint is number of flip bits, and thus not scaled
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    run_attack(attack_model, dataloader, criterion, constraint, args.mod_dim, args.pgd_steps, args.ext_lr, args.proj_lr, args.max_proj_iters, args.dataset=="modelnet", args.device)