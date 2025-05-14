import os
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import get_mnist_functa, get_mnist_loader
from utils import adjust_learning_rate, set_random_seeds, get_accuracy, Average
from tqdm import tqdm
import numpy as np
import argparse
from SIREN import ModulatedSIREN
from train_classifier import Classifier
from matplotlib import pyplot as plt
from higher import get_diff_optim
from autoattack import AutoAttack

class FullPGD(nn.Module):
    def __init__(self, inr, classifier, inner_steps = 100, inner_lr = 0.01, device='cuda'):
        """
        :param inr: pretrained inr model
        :param classifier: 'clean' pretrained classifier model to attack
        :param inner_steps: number of modulation optimization steps.
        :param inner_lr: learn rate for internal modulation optimization.
        :param device: use cuda or cpu.
        """
        super(FullPGD, self).__init__()
        #load classifier
        self.classifier = classifier.to(device)
        self.classifier.eval()
        
        #load inr
        self.inr = inr.to(device)
        self.inr.eval()
        
        #optimization params
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
       
    
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
            image = image[0].view(1, -1).T.to(device)
            
            inner_optimizer = torch.optim.SGD([modulator],lr=self.inner_lr) if clean else get_diff_optim(optim.SGD([modulator], lr=self.inner_lr), [modulator], device='cuda')
            
            mse = 0
            
            for step in range(self.inner_steps):
                
                if clean:
                    inner_optimizer.zero_grad()
                    fitted = self.inr(modulator)
                    inner_loss = inner_criterion(fitted.flatten(), image.flatten())
                    mse = inner_loss.item()
                    inner_loss.backward()
                    # Clip the gradient.
                    torch.nn.utils.clip_grad_norm_([modulator], 1)
                    # Update.
                    inner_optimizer.step()
                    
                    
                else:
                    fitted = self.inr(modulator)
                    inner_loss = inner_criterion(fitted.flatten(), image.flatten())
                    mse = inner_loss.item()
                    modulator, = inner_optimizer.step(inner_loss,params=[modulator])
                
         
            mses.append(mse)
            mods = modulator.unsqueeze(0) if mods is None else torch.cat((modulator.unsqueeze(0),mods),axis=0)
      
      
        return torch.flip(mods,dims=[0]) if not return_mse else (torch.flip(mods,dims=[0]), mse)
        
    
    
    def forward(self, x, start_mod = None, clean=False, return_mse = False):
        '''find modulation for x and classify it.
           x is input in signal domain, start_mod is optional starting point for modulation optimization, clean indicates if x is under clean (non-perturbed) evaluation 
           (so, if not, we can skip costly gradient tracking in modulation optimization), return_mse indicates whether to return the best modulation representation error.'''
        with torch.enable_grad(): #turn grad tracking on (Auto-Attack shuts it off in its clean evaluation)
            
            modulations = self.fit_image(x, start_mod,clean, return_mse)
           
            if return_mse:
                modulations,mse = modulations
         
            preds = self.classifier(modulations)
            if return_mse:
                return preds, modulations.detach(), mse
            else:
                return preds, modulations.detach()


    


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
    parser.add_argument('--batch-size', type=int, default=24, help = 'Size of minibatch')
    parser.add_argument('--depth', type=int, default=10, help='SIREN depth')
    parser.add_argument('--dataset', choices=["mnist", "fmnist"], help="Train for MNIST or Fashion-MNIST") #We currently do not support Full-PGD for ModelNet10
    parser.add_argument('--data-path', type=str, default='..', help='path to MNIST or FMNIST dataset')
    parser.add_argument('--siren-checkpoint', type=str, help='path to pretrained SIREN from meta-optimization')
    parser.add_argument('--classifier-checkpoint', type=str, help='path to pretrained classifier')
    parser.add_argument('--epsilon', type=int, default=16, help='attack epsilon -- epsilon/255 is the de-facto attack l_inf constraint.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    return parser.parse_args()
    
if __name__ == '__main__':

    
    args = get_args()    
    set_random_seeds(args.seed, args.device)
    device = args.device 
    dataloader = get_mnist_loader(args.data_path, train=False, batch_size=1, fashion = args.dataset=="fmnist")
    
    #Initiallize pretrained models
    modSiren = ModulatedSIREN(height=28, width=28, hidden_features=args.hidden_dim, num_layers=args.depth, modul_features=args.mod_dim) #28,28 is mnist and fmnist dims
    pretrained = torch.load(args.siren_checkpoint)
    modSiren.load_state_dict(pretrained['state_dict'])
    
    classifier = Classifier(width=args.cwidth, depth=args.cdepth,
                            in_features=args.mod_dim, num_classes=10).to(args.device)
    pretrained = torch.load(args.classifier_checkpoint)
    classifier.load_state_dict(pretrained['state_dict'])
    classifier.eval()
        
    attack_model = FullPGD(modSiren, classifier, inner_steps=args.mod_steps, inner_lr = args.inner_lr, device=args.device) #We do Auto-Attack and not Full-PGD - this is just for the internal optimization loop.
    attack_model.to(args.device)
        
    attack_model.eval()  

    
    eps = args.epsilon / 255
    # Wrap model forward for AutoAttack
    def wrapped_model(x):
        
        x = x.unsqueeze(1)
        logits = attack_model(x, clean=False)[0]
        return logits
    # Instantiate AutoAttack
    adversary = AutoAttack(
        model=wrapped_model,
        norm='Linf',
        eps=eps,
        version='standard',
        log_path=None
    )

   

    

    # Collect all data for AutoAttack
    all_x = []
    all_y = []
    for x, y in tqdm(dataloader, desc="Loading Data"):
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x[:4]).to(device)
    all_y = torch.cat(all_y[:4]).to(device)

    # Run AutoAttack
    with torch.no_grad():
        
        adv_complete,ylabels = adversary.run_standard_evaluation(all_x, all_y, bs=args.batch_size, return_labels = True)

    # Evaluate final accuracy
    
    with torch.no_grad():
        logits, _ = attack_model(adv_complete.unsqueeze(1), clean=True) #Here we can pass clean=True - we just evaluate and do not need 2nd-order gradients
        final_preds = logits.argmax(1)
        acc = (final_preds == all_y).float().mean().item()
        print(f"Final robust accuracy: {acc * 100:.2f}%") #Note - robust accuracy reported by Auto-Attack might lower - this is because this accuracy is evaluated in the differentiable approximation of the clean model.