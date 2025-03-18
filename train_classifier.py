import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_mnist_functa
from dataloader_modelnet import get_modelnet_functa
from utils import adjust_learning_rate, set_random_seeds, get_accuracy, Average
from tqdm import tqdm
import numpy as np
import argparse


# A classifier of MLP structure.
class Classifier(nn.Module):
    def __init__(self, width=1024, depth=3, in_features=512, num_classes=10, dropout=0.20, batchnorm=False):
        """
        :param width: number of neurons in hidden layers.
        :param depth:  number of hidden layers.
        :param in_features: number of input features.
        :param num_classes: number of classes.
        :param dropout: dropout probability.
        :param batchnorm: whether to include Batch-nomralization
        """
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.net = self._make_layers(batchnorm)

    def _make_layers(self,batchnorm=False):
        num_features = [self.in_features] + [self.width] * self.depth + [self.num_classes]
        layers = []
        for i in range(self.depth):
            layers.append(nn.Dropout(p=self.dropout))
            if batchnorm:
                
                layers.append(nn.BatchNorm1d(self.in_features))
            layers.append(nn.Linear(num_features[i], num_features[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_features[self.depth], num_features[self.depth + 1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_classifier(model, train_loader, optimizer, criterion, epoch):
    """
    :param model:
    :param train_loader:
    :param optimizer:
    :param criterion:
    :param epoch:
    :return: Loss.
    """
    model.train()
    device = next(iter(model.parameters())).device
    losses = []
    train_score = 0
    prog_bar = tqdm(train_loader, total=len(train_loader))
    for images, labels in prog_bar:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_score += preds.argmax(dim=-1).eq(labels).sum().item()
    accuracy = train_score / len(train_loader.dataset)
    print('epoch: %d, loss: %.4f, train acc: %.3f%s' % (
        epoch, sum(losses) / len(losses), accuracy * 100, '%'
    ))
    return losses


def eval_classifier(model, val_loader, epoch):
    """
    :param model:
    :param val_loader:
    :param epoch:
    :return: Validation Accuracy.
    """
    model.eval()
    device = next(iter(model.parameters())).device
    prog_bar = tqdm(val_loader, total=len(val_loader))
    top1acc = Average()
    top5acc = Average()

    for images, labels in prog_bar:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        top1acc_batch, top5acc_batch = get_accuracy(preds, labels, top_k=(1, 5))
        top1acc.update(top1acc_batch, labels.size(0))
        top5acc.update(top5acc_batch, labels.size(0))

    print('epoch: %d, val accuracy: top1 %.2f%s, top5 %.2f%s' % (
        epoch, top1acc.avg, '%', top5acc.avg, '%'
    ))
    return top1acc.avg, top5acc.avg

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--lr', type=float, default=0.01, help='classifier optimization lr')
    parser.add_argument('--cwidth', type=int, default=512, help='classifier MLP hidden dimension')
    parser.add_argument('--mod-dim', type=int, default=512, help='modulation dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate introduced in classifier optimization')
    parser.add_argument('--cdepth', type=int, default=3, help='classifier MLP depth')
    parser.add_argument('--batch-size',type=int,default=256, help='optimization mini-batch size')
    parser.add_argument('--dataset', choices=["mnist", "fmnist","modelnet"], help="Train for MNIST or Fashion-MNIST or ModelNet10")
    parser.add_argument('--num-epochs', type=int, default=160, help='number of classifier training epochs')
    parser.add_argument('--data-path', type=str, default='..', help='path to MNIST or FMNIST dataset')
    parser.add_argument('--functaset-path-train', type=str, help='path to optimized training Functaset')
    parser.add_argument('--functaset-path-test', type=str, help='path to optimized test Functaset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    return parser.parse_args()
if __name__ == '__main__':
    
    args = get_args()
    set_random_seeds(args.seed,args.device)
    
    
    loader_fn = get_modelnet_functa if args.dataset == "modelnet" else get_mnist_functa  
    train_functaloader = loader_fn(data_dir=args.functaset_path_train,mode='train', batch_size=args.batch_size)
    test_functaloader = loader_fn(data_dir=args.functaset_path_test,mode='test') #or mode=val

    

    # Load model and optimizer.
    classifier = Classifier(width=args.cwidth, depth=args.cdepth,
                            in_features=args.mod_dim, num_classes=10, dropout=args.dropout, batchnorm = args.dataset=="modelnet").to(args.device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(args.device)
    # Training.
    train_losses = []
    val_top1accs = []
    val_top5accs = []
    best_accuracy = 0
    model_dir = f"{args.dataset}_classifier"
    os.makedirs(model_dir,exist_ok=True)
    
    for epoch in range(args.num_epochs):
        if not args.dataset=="modelnet":
            adjust_learning_rate(optimizer, epoch, args.lr, args.num_epochs)
        elif epoch>=100 and not epoch%10:
            optimizer.param_groups[0]['lr'] /= 2
        losses_epo = train_classifier(
            model=classifier,
            train_loader=train_functaloader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
        )
        train_losses.extend(losses_epo)
        top1acc, top5acc = eval_classifier(
            model=classifier,
            val_loader=test_functaloader,
            epoch=epoch,
        )
        val_top1accs.append(top1acc)
        val_top5accs.append(top5acc)
        # Save the best model.
        if best_accuracy < top1acc:
            best_accuracy = top1acc
            torch.save({
                'epoch': epoch,
                'state_dict': classifier.state_dict(),
                'accuracy': best_accuracy,
            }, osp.join(model_dir, 'best_classifier.pth'))
    print(f"Best Accuracy: {best_accuracy:.2f} %")
    np.save(osp.join(model_dir, 'classifier_loss.npy'), np.array(train_losses))
    np.save(osp.join(model_dir, 'classifier_acc.npy'), np.array((val_top1accs, val_top5accs)))
