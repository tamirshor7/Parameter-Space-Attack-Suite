import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_mnist_loader
from dataloader_modelnet import get_modelnet_loader
from SIREN import ModulatedSIREN, ModulatedSIREN3D
from utils import adjust_learning_rate
from tqdm import tqdm
import os
import argparse
from utils import set_random_seeds

def fit(
        model,
        data_loader,
        outer_optimizer,
        outer_criterion,
        epoch_id,
        inner_steps=3,
        inner_lr=0.01,
        voxels=False,
):
    """
    Fit the INR for each specific sample for inner_steps steps to perform meta-learning.
    :param model: Meta-network INR.
    :param data_loader: Dataloader for dataset to train on.
    :param outer_optimizer: Meta-learning optimizer.
    :param outer_criterion: Meta-learning training objective.
    :param epoch_id: Epoch number.
    :param inner_steps: Number of internal, per-sample optimization steps for INR optimization.
    :param inner_lr: Learn-rate for internal, per-sample optimization.
    :param voxels: whether to use 3d data (.e.g modelnet) or 2d
    :return: Average representation loss.
    """
  
    losses = []
    device = next(iter(model.parameters())).device
    modul_features = model.modul_features
    inner_criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    prog_bar = tqdm(data_loader, total=len(data_loader))
    for images, labels in prog_bar:
        batch_size = images.size(0)
        images = (images.squeeze() if voxels else images.view(batch_size, 1, -1).moveaxis(1, -1)).to(device)
        modulators = []
        # Inner loop.
        for batch_id in range(batch_size):
            modulator = torch.zeros(modul_features).float().to(device)
            modulator.requires_grad=True
            inner_optimizer = (optim.Adam if voxels else optim.SGD)([modulator], lr=inner_lr)
            # Inner Optimization.
           
            for step in range(inner_steps):
                # Inner optimizer step.
                inner_optimizer.zero_grad()
                fitted = model(modulator)
           
                inner_loss = inner_criterion(fitted.T, images[batch_id].flatten()[None]) if voxels else inner_criterion(fitted, images[batch_id])
                inner_loss.backward()
             
                # Update.
                inner_optimizer.step()
            modulator.requires_grad = False
            modulators.append(modulator)

        outer_optimizer.zero_grad()
        outer_loss = torch.tensor(0).to(device).float()
        for batch_id in range(batch_size):
            modulator = modulators[batch_id]
            # Outer Optimization.
            fitted = model(modulator)
            outer_loss += (outer_criterion(fitted.T, images[batch_id].flatten()[None]) if voxels else outer_criterion(fitted, images[batch_id])) / batch_size
        # Outer optimizer step.
        outer_loss.backward()
        # Clip the gradient.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        outer_optimizer.step()
        losses.append(outer_loss.item())
        

        prog_bar.set_description(desc='Epoch {}, loss {:.6f}'.format(epoch_id, outer_loss.item()))

    print(f'epoch: {epoch_id}, loss: {sum(losses)/ len(losses)}')
    return sum(losses) / len(losses)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--ext-lr', type=float, default=5e-6, help='external optimization loop lr')
    parser.add_argument('--int-lr', type=float, default=0.01, help='internal optimization loop lr')
    parser.add_argument('--batch-size', type=int, default=128, help='optimization minibatch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='SIREN hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--depth', type=int, default=10, help='SIREN depth')
    parser.add_argument('--dataset', choices=["mnist", "fmnist","modelnet"], help="Train for MNIST or Fashion-MNIST or ModelNet10")
    parser.add_argument('--num-epochs', type=int, default=6, help='number of epochs for external optimization')
    parser.add_argument('--data-path', type=str, default='..', help='path to MNIST, FMNIST or ModelNet10 dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    return parser.parse_args()

if __name__ == '__main__':
    # Training Parameters.
  
    args = get_args()
    
    device = args.device
    set_random_seeds(args.seed,device)
    if args.dataset == "modelnet":
        resample_shape = (15,15,15) #we use this resampling in all experiments
        dataloader = get_modelnet_loader(train=True, batch_size=args.batch_size, resample_shape=resample_shape)
        modSiren = ModulatedSIREN3D(height=resample_shape[0], width=resample_shape[1], depth=resample_shape[2],\
            hidden_features=args.hidden_dim, num_layers=args.depth, modul_features=args.mod_dim) #we use a mod dim of 2048 in our exps
  
    else:        
        dataloader = get_mnist_loader(args.data_path, train=True, batch_size=args.batch_size, fashion = args.dataset=="fmnist")
        modSiren = ModulatedSIREN(height=28, width=28, hidden_features=args.hidden_dim, num_layers=args.depth, modul_features=args.mod_dim) #28,28 is mnist and fmnist dims

        
      
    
    modSiren = modSiren.to(args.device)
    optimizer = optim.Adam(modSiren.parameters(), lr=args.ext_lr)
    criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    
    savedir = f"model_{args.dataset}"
    
    os.makedirs(savedir, exist_ok=True)
    best_loss = float('Inf')
    for epoch in range(args.num_epochs):
        loss = fit(
            modSiren, dataloader, optimizer, criterion, epoch, inner_steps=3,inner_lr=args.int_lr, voxels=args.dataset=='modelnet'
        )
        if loss < best_loss:
            best_loss = loss
            torch.save({'epoch': epoch,
                        'state_dict': modSiren.state_dict(),
                        'loss': best_loss,
                        }, f'{savedir}/modSiren.pth')

