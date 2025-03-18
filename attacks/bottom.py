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


class BOTTOM(nn.Module):
    def __init__(self, inr, classifier, bottom_steps=5, inner_steps = 100, inner_lr = 0.01, voxels = False, device='cuda'):
        """
        :param inr: pretrained inr model
        :param classifier: 'clean' pretrained classifier model to attack
        :param bottom_steps: Number of modulation optimization steps through which we differentiate for modulation optimization. 
        :param inner_steps: number of modulation optimization steps in full (non-truncated) optimization.
        :param inner_lr: learn rate for internal modulation optimization.
        :param: voxels: use 3d data
        :param device: use cuda or cpu.
        """
        super(BOTTOM, self).__init__()
        #load classifier
        self.classifier = classifier.to(device)
        self.classifier.eval()
        
        #load inr
        self.inr = inr.to(device)
        self.inr.eval()
        
        #optimization params
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.bottom_steps = bottom_steps
        self.inner_optimizer_fn = optim.Adam if voxels else optim.SGD
        
        self.voxels = voxels
        
       
    
    def fit_image(self, x, start_mod, clean, return_mse = False):
        '''optimize and return modulation for a specific (potentially perturbed) signal-domain input x'''
        device = "cuda"
        inner_criterion = nn.MSELoss().cuda()
        
        
        mses = []
        mods = None
        # Inner Optimization.
        for image in x:
            modulator = start_mod.squeeze().detach().clone().float().to(device) if start_mod is not None else torch.zeros(self.inr.modul_features).float().to(device)
         
            modulator.requires_grad = True
            if not self.voxels:
                image = image[0]
            image = image.view(1, -1).T.to(device)
            
            inner_optimizer = self.inner_optimizer_fn([modulator],lr=self.inner_lr) if clean else get_diff_optim(self.inner_optimizer_fn([modulator], lr=self.inner_lr), [modulator], device='cuda')
           
            mse = 0
            for step in range(self.inner_steps if clean else self.bottom_steps):
                
                if clean:
                    inner_optimizer.zero_grad()
                    fitted = self.inr(modulator)
                    
                    inner_loss = inner_criterion(fitted.T, image.flatten()[None]) if self.voxels else inner_criterion(fitted.flatten(), image.flatten())
                    mse = inner_loss.item()
                    inner_loss.backward()
                    # Clip the gradient.
                    torch.nn.utils.clip_grad_norm_([modulator], 1)
                    # Update.
                    inner_optimizer.step()
                    
                    
                else:
                    fitted = self.inr(modulator)
                    inner_loss = inner_criterion(fitted.T, image.flatten()[None]) if self.voxels else inner_criterion(fitted.flatten(), image.flatten())
                    mse = inner_loss.item()
                    modulator, = inner_optimizer.step(inner_loss,params=[modulator])
                
        
            mses.append(mse)
            mods = modulator.unsqueeze(0) if mods is None else torch.cat((modulator.unsqueeze(0),mods),axis=0)
      
        
        return torch.flip(mods,dims=[0]) if not return_mse else (torch.flip(mods,dims=[0]), mse)
        
    
    
    def forward(self, x, start_mod = None, clean=False, return_mse = False):
        '''find modulation for x and classify it.
           x is input in signal domain, start_mod is optional starting point for modulation optimization, clean indicates if x is under clean (non-perturbed) evaluation 
           (so, if not, we can skip costly gradient tracking in modulation optimization), return_mse indicates whether to return the best modulation representation error.'''
    
        modulations = self.fit_image(x, start_mod,clean, return_mse)
        if return_mse:
            modulations,mse = modulations
     
        preds = self.classifier(modulations)
        if return_mse:
            return preds, modulations.detach(), mse
        else:
            return preds, modulations.detach()


    
def run_attack(model, loader, criterion, constraint, pgd_iters, pgd_lr, voxels = False, device='cuda'):
   
    """
    :param model: attack model holding pretrained INR and classifier.
    :param loader: signal domain data-loader for data for which we optimize attacks.
    :param criterion: clean classifier optimization criterion.
    :param constraint: l_inf attack bound.
    :param pgd_iters: number of PGD iterations.
    :param pgd_lr: learn-rate for PGD attack.
    :param voxels: use voxel-grid data if True
    :param device: use cuda or cpu.
    """
    prog_bar = tqdm(loader, total=len(loader))
    acc_sum = 0
    samples_num = 0
    rights_clean = 0
    rights_attacked = 0
    samples=0
    
    for x,labels in prog_bar:
    
        samples += 1
        x = x.cuda()
        labels = labels.cuda()
        
        
        clean, base_mod, clean_mse = model(x.unsqueeze(1), clean=True, return_mse = True)
        rights_clean += (clean.argmax(1)==labels).item()
        labels = labels.cuda()
        
        if voxels:
            mask = BinaryMask(constraint).to(device)
        else:
            pert = torch.zeros(28,28).to(device)
            pert.requires_grad = True
            optimizer = torch.optim.Adam([pert],lr=pgd_lr)
       
        
        for iter in range(pgd_iters):
            for istep in range(model.inner_steps//model.bottom_steps):
                if voxels:
                    output, base_mod = model(mask(x).unsqueeze(1), base_mod,clean=False)         
                    loss = 1-criterion(output,labels)               
                    loss.backward()
                    mask.mask_grad(lr=pgd_lr)
                else:
                    
                    optimizer.zero_grad()
                    proj_pert = torch.clamp(pert, -constraint, constraint)
                    output, cur_mod = model((x+proj_pert).unsqueeze(1), base_mod, clean=False)
                  
                    loss = 1-criterion(output,labels)
                    
                    loss.backward()
               
                    optimizer.step()
                
              
                
            
          
    
        
        
        
        #Final Evaluation for perturbation
        if voxels:
            output, cur_mod, final_mse = model(mask(x).detach().unsqueeze(1), clean=True, return_mse = True)
        else:
            output, cur_mod, final_mse = model((x+proj_pert).detach().unsqueeze(1), clean=True, return_mse = True)
 
        print(f"Clean MSE: {clean_mse}. Perturbation Final MSE: {final_mse}")
        with torch.no_grad():
            
            attack_succ = (output.argmax(1) != labels).item()
            if not attack_succ:
                rights_attacked += 1
            
            
            
        prog_bar.set_description(f'Constraint {constraint}: Curr clean acc {rights_clean/samples}; Curr attacked acc {rights_attacked/samples}.')
        
    prog_bar.set_description(f'Constraint {constraint}: Final clean acc {rights_clean/samples}; Final attacked acc {rights_attacked/samples}.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--inner-lr', type=float, default=0.01, help='learn rate for internal modulation optimization')
    parser.add_argument('--ext-lr', type=float, default=0.01, help='learn rate for external adversarial perturbation optimization')
    parser.add_argument('--hidden-dim', type=int, default=256, help='SIREN hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--mod-steps', type=int, default=10, help='Number of internal modulation optimization steps per PGD iteration, without truncation (for clean evaluation)')
    parser.add_argument('--pgd-steps', type=int, default=100, help='Number of projected gradient descent steps')
    parser.add_argument('--interleave-steps', type=int, default=5, help='For each PGD iteration, calculate the modulation optimization gradients and backprop through every\
                                                                        interleave-steps internal modulation optimization iterations.')
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
    assert not args.mod_steps%args.interleave_steps #BOTTOM steps must be inetgerly intertwined within the entire optimization schedule for each PGD iteration
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
    
    attack_model = BOTTOM(modSiren, classifier, bottom_steps=args.interleave_steps, inner_steps=args.mod_steps, inner_lr = args.inner_lr, voxels = args.dataset=='modelnet',device=args.device)
    attack_model.to(args.device)
    constraint = int(args.epsilon) if args.dataset=="modelnet" else args.epsilon/255 #for the 3D case the constraint is number of flip bits, and thus not scaled
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    run_attack(attack_model, dataloader, criterion, constraint, args.pgd_steps, args.ext_lr, args.dataset=="modelnet", args.device)