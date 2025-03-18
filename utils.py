import numpy as np
import torch
import os
import random


# Reproducibility guarantee.
def set_random_seeds(seed=0, device='cuda:0'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Adjust learning rate in each epoch.
def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    lr = initial_lr
    if epoch >= 0.5 * num_epochs:
        lr *= 0.1
    if epoch >= 0.75 * num_epochs:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Get the top-k accuracy of classification.
def get_accuracy(yHat, yTrue, top_k=(1,)):
    max_k = max(top_k)
    batchSize = yTrue.size(0)

    # Get the class labels.
    _, pred = yHat.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(yTrue.view(1, -1).expand_as(pred))

    # Compute the accuracy.
    acc = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc.append(correct_k.mul_(100.0 / batchSize).item())
    return acc


# Recording and compute real-time average.
class Average:
    def __init__(self):
        self.cur = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur, batchSize):
        self.cur = cur
        self.sum += cur * batchSize
        self.count += batchSize
        self.avg = self.sum / self.count

#binary mask for binary voxelgrid perturbation optimization
class BinaryMask(torch.nn.Module):
    def __init__(self,constraint, device='cuda', shape=(15,15,15)): #constraint is in l_0
        super().__init__()
         
        self.cons = constraint 
       
        self.mask = torch.randn(*shape)[None].to(device)
        self.binary_mask = self.mask.clone().to(device)
        
        self.mask = torch.nn.Parameter(self.mask,requires_grad=False)
        self.binary_mask = torch.nn.Parameter(self.binary_mask,requires_grad=True)
         
    
    def apply_mask(self,x):

        drop_threshold = torch.topk(self.mask.flatten(), int(self.cons), largest=True).values.min()
        with torch.no_grad():
            self.binary_mask.data = (self.mask >= drop_threshold).to(torch.float)
        
        return x + self.binary_mask - 2*self.binary_mask*x #differentiable xor

    
    def mask_grad(self,lr, grad=None):
        
        if grad is None:
            grad = self.binary_mask.grad
        self.mask-= lr*grad
        self.mask.clamp(-1,1)
        
        return
    
    def forward(self, x):
        return self.apply_mask(x) 
        
        