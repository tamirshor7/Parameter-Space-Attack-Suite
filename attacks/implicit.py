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


class ImplicitArgmin(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_tensor, v_star, inner_lr, inner_steps, inr, dvdx):
        
        ctx.save_for_backward(input_tensor)
        ctx.inr = inr
        
        x = input_tensor.clone()
        ctx.v_star = v_star.detach()
        ctx.dvdx = dvdx
        return v_star
        
        
        
        

    @staticmethod
    def backward(ctx, grad_output):
       
        
        grad_input =ctx.dvdx@grad_output
        return grad_input, None, None, None, None, None


class ArgminLayer(nn.Module):
    def __init__(self, inner_steps, inner_lr, inr):
        super(ArgminLayer, self).__init__()
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.inr = inr

    def forward(self, x, return_mse = False, clean = False):
        #use the custom autograd function
        device = "cuda"
        inner_criterion = nn.MSELoss().cuda()
        
        x=x.clone().detach()
        mses = []
        mods = None
        #inner optimization
        for image in x:
            modulator = torch.zeros(self.inr.modul_features).float().to(device)
         
            modulator.requires_grad = True
            
            
            image = image[0].view(1, -1).T.to(device)
            
            def closure():
                inner_optimizer.zero_grad()
                fitted = self.inr(modulator)
                inner_loss = inner_criterion(fitted, image)
                inner_loss.backward()
                return inner_loss

            inner_optimizer = optim.LBFGS([modulator], lr=self.inner_lr, max_iter=self.inner_steps, line_search_fn="strong_wolfe")
            inner_optimizer.step(closure)
                   
            with torch.no_grad():
                mse = torch.nn.MSELoss()(self.inr(modulator),image).item() 
            
            mses.append(mse)
            mods = modulator.unsqueeze(0) if mods is None else torch.cat((modulator.unsqueeze(0),mods),axis=0)
      
        
        mods = torch.flip(mods,dims=[0])
        
        #compute gradient of inr w.r.t modulation
        
        mods = mods[0].detach()
        mods.requires_grad=True
        x=x.flatten()
        x.requires_grad = True
        mse = torch.nn.functional.mse_loss(self.inr(mods.flatten()).T,x.flatten()[None])
        if clean:
            prod = None
        else:
            dfdv = torch.autograd.grad(mse,mods,create_graph=True)[0]
            
            '''
            Compute the second mixed derivative d^2f/dxdy.
            Finally, compute the second derivative with respect to y (d^2f/dy^2)
            '''
            hessianyy = []
            hessianyx = []
            for d in dfdv:
                
                hessianyy.append(torch.autograd.grad(d[None], mods,retain_graph=True)[0][None])
                hessianyx.append(torch.autograd.grad(d[None], x,retain_graph=True)[0][None])
            hessianyy = torch.cat(hessianyy,0)
            hessianyx = torch.cat(hessianyx,0)
            
            prod = (-hessianyx.T@hessianyy.inverse()).T
            
        if return_mse:
            return (mods,mse), prod
        else:
            return mods, prod


        

class Implicit(nn.Module):

    def __init__(self, inr, classifier, inner_steps = 100, inner_lr = 0.01, device='cuda'):
        """
        :param inr: pretrained inr model
        :param classifier: 'clean' pretrained classifier model to attack
        :param inner_steps: number of modulation optimization steps in full (non-truncated) optimization.
        :param inner_lr: learn rate for internal modulation optimization.
        :param device: use cuda or cpu.
        """
        super(Implicit, self).__init__()
        #load classifier
        self.classifier = classifier.to(device)
        self.classifier.eval()
        
        #load inr
        self.inr = inr.to(device)
        self.inr.eval()
        
        #optimization params
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        self.argmin = ArgminLayer(inner_steps, inner_lr, self.inr)
    
    
    
    def forward(self, x, start_mod = None, clean=False, return_mse = False):
        '''
        Get optimized data modulation, and the gradient of the argmin operation dvdx, then classifiy modulations. 
        Namely, v are the modulation vectors and x is signal domain data, dvdx is partial derivation of v w.r.t x.
        dvdx is computed here only to be returned for first-order optimization later.
        '''
        modulations, dvdx = self.argmin(x,return_mse=return_mse, clean=clean)
        if return_mse:
            modulations,mse = modulations
        modulations = modulations.detach()
        modulations.requires_grad = True
        
        
        preds = self.classifier(modulations.reshape(1,-1)).squeeze()
        if return_mse:
            return preds, modulations.detach(),dvdx, mse
        else:
            return preds, modulations,dvdx

    

    
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
    projection_loss_eps = constraint/100 #for stop criteria of iterative projection
    
    for x,labels in prog_bar:
    
        samples += 1
        x = x.cuda()
        labels = labels.cuda()
        
        
        clean, clean_mod,dvdx,initial_mse = model(x.unsqueeze(1), return_mse = True)
        right_clean = (clean.argmax()==labels).item()
        rights_clean += right_clean
        
        if not right_clean:
                continue #skip perturbation optimization if classifier is already wrong, to save time.
        labels = labels.cuda()
        if voxels:
            mask = BinaryMask(constraint).to(device)
        else:
            pert = torch.zeros(28,28).to(device)
            pert.requires_grad = True
            optimizer = torch.optim.Adam([pert],lr=pgd_lr)
       
        losses = [] #Tracking for early-stop criteria
        
        for iter in range(pgd_iters):
            
            if voxels:
                with torch.no_grad():
                    perturbed_x = mask(x)[None]
                    dxdp = ((1-2*x)*mask.binary_mask).flatten() #simple manual grad computation - of perturbed signal domain data to perturbation
                    
                output, cur_mod,dvdx = model(perturbed_x, clean_mod, clean=False)
                output = output[None]
                with torch.no_grad():
                    if (output.argmax(1) != labels).item():
                        break
              
                loss = 1-criterion(output,labels)
                
                with torch.no_grad():
                    dldv = torch.autograd.grad(loss,cur_mod)[0] #grad of classification loss w.r.t modulation vectors
                    
                
                #Note we avoid forming the 15^3X15^3 diagonal matrix of dxdp, and instead perfrom equivalent elementwise multpilication.
                mask.mask_grad(lr=pgd_lr, grad=((dldv@dvdx)*dxdp).reshape(1,15,15,15)) 
                                                                                            
                losses.append(loss.item())
                
                if iter and not iter%1:
                   
                    print(f"Loss: {loss.item()}, dldv: {dldv.abs().mean().item()}, dvdx: {dvdx.abs().mean().item()}")
                
                if len(losses)>2 and np.abs(losses[-1]-losses[-2])<1e-6 and np.abs(losses[-2]-losses[-3])<1e-6 :
                    break
            else:
                optimizer.zero_grad()
                with torch.no_grad():
                    proj_pert = torch.clamp(pert, -constraint, constraint)
                    dppdp = torch.zeros(28**2,28**2).cuda() #dppdp is grad of projected perturbation w.r.t original perturbation ; 28X28 is 2-D data dimension
                    #compute dppdp values
                    mask = (pert.flatten() >= -constraint) & (pert.flatten() <= constraint)
                    dppdp[torch.arange(28**2), torch.arange(28**2)] = mask.float()
                
                output, cur_mod,dvdx = model((x+proj_pert).unsqueeze(1), clean_mod, clean=False)
                with torch.no_grad():
                    if (output.argmax() != labels).item():
                        break
                loss = 1-criterion(output[None],labels)
              
                with torch.no_grad():
                    dldv = torch.autograd.grad(loss,cur_mod)[0] #grad of classification loss w.r.t modulation vectors
                    pert.grad = (dppdp@dvdx@dldv).reshape(28,28) #manual gradient computation by the chain rule (under transpose)
                  
                optimizer.step()
                losses.append(loss.item())
                if len(losses)>2 and np.abs(losses[-1]-losses[-2])<1e-6 and np.abs(losses[-2]-losses[-3])<1e-6 :
                    break
          
    
        
        
        
        #Final Evaluation for perturbation
        if voxels:
            output, cur_mod, _, final_mse = model(mask(x).detach().unsqueeze(1), clean_mod.detach(), clean=True, return_mse = True)
        else:
            output, cur_mod, _, final_mse = model((x+proj_pert).detach().unsqueeze(1), None, clean=True, return_mse = True)
 
        
        with torch.no_grad():
            attack_succ = (output.argmax() != labels).item()
            if not attack_succ:
                rights_attacked += 1
            
            
            
        prog_bar.set_description(f'Constraint {constraint}: Curr clean acc {rights_clean/samples}; Curr attacked acc {rights_attacked/samples}.')
        
    prog_bar.set_description(f'Constraint {constraint}: Final clean acc {rights_clean/samples}; Final attacked acc {rights_attacked/samples}.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--inner-lr', type=float, default=0.1, help='learn rate for internal modulation optimization')
    parser.add_argument('--ext-lr', type=float, default=0.5, help='learn rate for external adversarial perturbation optimization')
    parser.add_argument('--hidden-dim', type=int, default=256, help='SIREN hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--mod-steps', type=int, default=50, help='Number of internal modulation optimization steps per PGD iteration, without truncation (for clean evaluation)')
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
    
    attack_model = Implicit(modSiren, classifier, inner_steps=args.mod_steps, inner_lr = args.inner_lr,device=args.device)
    attack_model.to(args.device)
    constraint = int(args.epsilon) if args.dataset=="modelnet" else args.epsilon/255 #for the 3D case the constraint is number of flip bits, and thus not scaled
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    run_attack(attack_model, dataloader, criterion, constraint, args.pgd_steps, args.ext_lr, args.dataset=="modelnet", args.device)