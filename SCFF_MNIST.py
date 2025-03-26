import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import ExponentialLR, StepLR, LinearLR

import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt

import torchvision
from torchvision.transforms import transforms, ToPILImage, Compose, ToTensor,RandomAffine, Normalize, Lambda
from torchvision.datasets import MNIST

from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split,Subset
import argparse
import time

import numpy as np
from numpy import fft 
import math
import optuna
import json



def get_arguments(): 
    
    parser = argparse.ArgumentParser(description="Pretrain a CNN using SCFF", add_help=False)

    parser.add_argument("--epochs", type=int, default=20,
                        help='Number of epochs')
    parser.add_argument("--tr_and_eval", action='store_true',
                        help='train while evaluating')
    parser.add_argument('--device_num',type=int, default=0,
                        help='device to use for training / testing')
    parser.add_argument("--save_model", action='store_true',
                        help='save model or not')
    parser.add_argument("--NL", type=int, default=1,
                        help='Number of layers')
    return parser


class AugmentedMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, augment_transform_1=None, 
                 augment_transform_2=None, target_transform=None, download=False):
        super(AugmentedMNIST, self).__init__(root, train=train, transform=transform, 
                                             target_transform=target_transform, download=download)
        self.augment_transform_1 = augment_transform_1
        self.augment_transform_2 = augment_transform_2

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # Convert image to PIL Image for transformation
        img = ToPILImage()(img)

        # Apply the original transform
        if self.transform is not None:
            orig_img = self.transform(img)

        # Apply the first augmented transform
        if self.augment_transform_1 is not None:
            aug_img_1 = self.augment_transform_1(img)
        else:
            aug_img_1 = self.transform(img)

        # Apply the second augmented transform
        if self.augment_transform_2 is not None:
            aug_img_2 = self.augment_transform_2(img)
        else:
            aug_img_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return orig_img, aug_img_1, aug_img_2, target

class CustomMNIST(MNIST):
    def __init__(self, root, train=True, transform=None,download=False):
        super(CustomMNIST, self).__init__(root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # Convert image to PIL Image for transformation
        img = ToPILImage()(img)

        # Apply the original transform
        if self.transform is not None:
            orig_img = self.transform(img)

        return orig_img, target

def get_train(batchsize, augment):

    torch.manual_seed(42)
    # Transformation pipeline
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x))])

    transform_tr = Compose([
        RandomAffine(degrees=0, translate=(2/28, 2/28)),
        ToTensor(),
        Lambda(lambda x: torch.flatten(x))
        ])

    if augment:
        trainset = AugmentedMNIST(root='data', train=True, download=True, transform=transform, 
                                augment_transform_1=transform_tr, augment_transform_2=transform_tr)
    else:
        trainset = AugmentedMNIST(root='data', train=True, download=True, transform=transform, 
                                augment_transform_1=None, augment_transform_2=None)
    #mnist_train = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_tr)
    mnist_test = torchvision.datasets.MNIST(root='data', download=True, train=False, transform=transform)

    sup_trainset = CustomMNIST(root='data',transform=transform, train=True, download=True)

    train_size = 60000
    val_size = 0

    indices = torch.randperm(len(trainset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]

    # Create subsets
    mnist_train = Subset(trainset, train_indices)
    sup_train_data = Subset(sup_trainset, train_indices)
    mnist_val = Subset(trainset, val_indices)

    train_loader = DataLoader(mnist_train, batch_size= batchsize, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size= batchsize, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size= 1000, shuffle=False)
    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader, sup_trainloader


def get_pos_neg_batch_imgcats(batch_pos1, batch_pos2, p = 1):
    """
    Generates positive and negative inputs for SCFF.

    Args:
        batch_pos1 (torch.Tensor): First set of samples of shape (batch_size, ...).
        batch_pos2 (torch.Tensor): Second set of samples, typically an augmented version 
                                   of batch_pos1 with the same shape or the same with batch_pos1.
        p (int, optional): Number of negative samples per positive sample. Default is 1.

    Returns:
        tuple: 
            - batch_pos (torch.Tensor): Concatenated positive samples of shape (batch_size, 2 * feature_dim).
            - batch_negs (torch.Tensor): Concatenated negative samples of shape (batch_size * p, 2 * feature_dim).
    """
    batch_size = len(batch_pos1)

    batch_pos =torch.cat((batch_pos1, batch_pos2), dim = 1)

    #create negative samples
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p,batch_size - 1)]
    labeles = torch.arange(batch_size)

    batch_negs = []
    for i in random_indices:
        batch_neg = batch_pos2[(labeles+i)%batch_size]
        batch_neg = torch.cat((batch_pos1, batch_neg), dim = 1)
        batch_negs.append(batch_neg)
    
    return batch_pos, torch.cat(batch_negs)



def stdnorm (x, dims = [1,2,3]):

    x = x - torch.mean(x, dim=(dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(dims), keepdim=True))

    return x

def l2norm (x):

    x = x/(x.norm(2, 1, keepdim=True) + + 1e-10)
    
    return x
    
class standardnorm(nn.Module):
    def __init__(self, dims = [1,2,3]):
        super(standardnorm, self).__init__()
        self.dims = dims

    def forward(self, x):
        x = x - torch.mean(x, dim=(self.dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(self.dims), keepdim=True))
        return x

class L2norm(nn.Module):
    def __init__(self, dims = [1,2,3]):
        super(L2norm, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x / (x.norm(p=2, dim=(self.dims), keepdim=True) + 1e-10)

class triangle(nn.Module):
    def __init__(self):
        super(triangle, self).__init__()

    def forward(self, x):
        x = x - torch.mean(x, axis=1, keepdims=True)
        return F.relu(x)

class Layer(nn.Linear):
    """
    A custom fully connected (linear) layer with optional normalization, activation functions
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        Norm (str): Normalization method, either "L2norm" or "stdnorm".
        droupout (float, optional): Dropout rate. Default is 0.0 (no dropout).
        act (int, optional): Activation function choice (0 for ReLU, 1 for Triangle). Default is 0.
        bias (bool, optional): If True, includes bias in linear transformation. Default is True.
        concat (bool, optional): Whether to split input channels and apply convolution separately before summing. Default is True.
        device (str, optional): Specifies the computation device (CPU or CUDA). Abandon.
    """
    def __init__(self, in_features, out_features, Norm, droupout = 0.0, act = 0, bias=True, concat = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        if act == 0:
            self.act = torch.nn.ReLU()
        else: #elif act == 1
            self.act = triangle()
        
        self.relu = torch.nn.ReLU()
        self.concat = concat
        
        if Norm == "L2norm":
            self.norm = L2norm(dims = [1])
        else:
            self.norm = standardnorm(dims = [1])

    def forward(self, x):
        """
        Forward pass of the custom Layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Transformed output tensor of shape (batch_size, out_features).
        """
        x_direction =self.norm(x)
        if self.concat:
            lenchannel = x.size(1)//2
            x = torch.mm(x_direction[:,:lenchannel], self.weight.T
                         ) + torch.mm(x_direction[:,lenchannel:], self.weight.T) + 2*self.bias
        else:
            x = torch.mm(x_direction, self.weight.T) + self.bias
        return x



def train(nets, device, optimizers,schedulers, threshold1,threshold2, epochs
            , a,b, lamda, freezelayer,period,tr_and_eval, Layer_out,trainloader
            , valloader, testloader, suptrloader, out_dropout,
            search,p,pre_std,norm,aug):
    """
    Trains a set of neural networks using SCFF.

    Args:
        nets (list): List of neural network models.
        device (str): Device ('cuda' or 'cpu') to run the training on.
        optimizers (list): List of optimizers for each network.
        schedulers (list): List of learning rate schedulers.
        threshold1 (list): List of Threshold values for Positive examples. 
        threshold2 (list): List of Threshold values for negative examples.
        epochs (int): Number of training epochs.
        a (float): Scaling factor for positive sample loss, default=1.
        b (float): Scaling factor for negative sample loss, default=1.
        lamda (list): Regularization coefficients.
        freezelayer (int):  Number of layers where weights were frozon.
        period (list): Number of batches before updating the learning rate.
        tr_and_eval (bool): Whether to evaluate the model during training.
        Layer_out (list): Layers to use for final classifier.
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        testloader (DataLoader): DataLoader for testing data.
        suptrloader (DataLoader): DataLoader for supervised training data.
        out_dropout (float): Dropout rate for classification layers.
        search (bool): Whether to use validation or test set for evaluation.
        p (int): Number of negative samples per positive sample.
        pre_std (bool): Whether to standardize input before training.
        norm (callable): Normalization function for feature extraction on the extracted feature for evaluation.
        aug (int): Augmentation mode (0: original, 1: single, 2: dual).

    Returns:
        tuple: (trained models, positive sample losses, negative sample losses, feature dimensions, evaluation results).
    """

    # Store goodness for positive and negative samples
    all_pos = []
    all_neg = []
    NL = len(nets)
    
    for _ in range(NL):
        all_pos.append([])
        all_neg.append([])
        
    firstpass=True
    nbbatches = 0

    NBLEARNINGEPOCHS = epochs

    if epochs == 0:
        N_all = NBLEARNINGEPOCHS + 1
    else:
        N_all = NBLEARNINGEPOCHS

    Dims = []
    taccs = []
    # Start the experiment !
    for epoch in range(N_all):

        print("Epoch", epoch)
        #correct=0; total=0
        if epoch < NBLEARNINGEPOCHS and epochs !=0:
            nets[-1].train()
            print("Unlabeled.")
            UNLAB = True; 
            zeloader = trainloader
        else: # evaluate the output neurons without train (freezelayer = NL)
            print("Evaluate the trained features.")
            for net in nets:
                net.eval()
            if epoch == NBLEARNINGEPOCHS:
                # With frozen weights, acquire network responses to training set
                UNLAB = False; 
                zeloader = testloader
            else:
                raise(ValueError("Wrong epoch!")) 


        goodness_pos = 0
        goodness_neg = 0

        for numbatch, (x,x_aug1,x_aug2, targets) in enumerate(zeloader):

            nbbatches += 1

            x = x.to(device)
            x_aug1 = x_aug1.to(device)
            x_aug2 = x_aug2.to(device)
            
            if pre_std:
                x = stdnorm(x, dims = [1])
                x_aug1 = stdnorm(x_aug1, dims = [1])
                x_aug2 = stdnorm(x_aug2, dims = [1])

            if aug == 0:
                x, x_neg = get_pos_neg_batch_imgcats(x, x, p= p)
            elif aug == 1:
                x, x_neg = get_pos_neg_batch_imgcats(x_aug1, x_aug1, p= p)
            else:
                x, x_neg = get_pos_neg_batch_imgcats(x_aug1, x_aug2, p= p)
                    

            for i in range(NL):

                optimizers[i].zero_grad()
                
                x = nets[i](x)
                x_neg = nets[i](x_neg)

                yforgrad = nets[i].relu(x).pow(2).mean([1])
                yforgrad_neg =nets[i].relu(x_neg).pow(2).mean([1])

                #print(yforgrad.mean([0]), yforgrad_neg.mean([0]))
                
                if i < freezelayer:
                    UNLAB = False
                else:
                    UNLAB = True

                if UNLAB :
                    loss1 =  torch.log(1 + torch.exp(
                        a*(- yforgrad  + threshold1[i]))).mean(
                        ) + torch.log(1 + torch.exp(
                            b*(yforgrad_neg  - threshold2[i]))).mean(
                            ) +  lamda[i] * torch.norm(yforgrad[:,None], p=2, dim = (1)).mean() #+ lamda2[i]*(F.relu(yita - x.std(dim = 0)).mean()) + lambda_covar*covar_reg(x
                            

                    loss1.backward()
                    optimizers[i].step()  

                    if (nbbatches+1)%period[i] == 0:
                        schedulers[i].step()
                        print(f'nbbatches {nbbatches+1} learning rate: {schedulers[i].get_last_lr()[0]}')  

                x = nets[i].act(x).detach()
                x_neg = nets[i].act(x_neg).detach()

                if firstpass:
                    print("Layer", i, ": x.shape:", x.shape, "y.shape (after MaxP):", x.shape, end=" ")
                    _, w = x.shape
                    Dims.append(w)
            
            firstpass = False
            goodness_pos += (torch.mean(yforgrad)).item()
            goodness_neg += (torch.mean(yforgrad_neg)).item()

            if UNLAB and numbatch == len(zeloader) - 1:
                print(goodness_pos, goodness_neg)
                all_pos[i].append(goodness_pos)
                all_neg[i].append(goodness_neg)
                goodness_pos,  goodness_neg = 0, 0
        
        if tr_and_eval:
            if epoch>3 and epoch%1==0:
                tacc = evaluation_(nets, Dims, device, Layer_out, out_dropout
                                   ,search, valloader, testloader, suptrloader, pre_std, norm)

                taccs.append(tacc)

    print("Training done..")
    if tr_and_eval:
        return nets, all_pos, all_neg, Dims, taccs
    else:
        return nets, all_pos, all_neg, Dims

class CustomStepLR(StepLR):
    """
    Custom Learning Rate schedule with step functions for supervised training of linear readout (classifier)
    """

    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


def process_batch(nets, x, Layer, norm):
    outputs = []
    with torch.no_grad():
        for j, net in enumerate(nets):
            x = net(x)
            x = net.act(x).detach()
            out = norm(x)
            if j in Layer:
                outputs.append(out)

    return torch.cat(outputs, dim=1)

def evaluation_(nets, dims, device, Layer_out, out_dropout, search, valloader, tsloader, suptrloader, pre_std, norm):
    """
    Evaluates a neural network model by training a classifier on the learned features and testing on validation/test data.

    Args:
        nets (list): List of trained networks.
        dims (list): List of output dimensions for each layer.
        device (str): Device to use ('cuda' or 'cpu').
        Layer_out (list): Indices of layers to extract features from.
        out_dropout (float): Dropout probability for the classifier.
        search (bool): If True, uses the validation set, otherwise uses the test set.
        valloader (DataLoader): DataLoader for validation set.
        tsloader (DataLoader): DataLoader for test set.
        suptrloader (DataLoader): DataLoader for supervised training.
        pre_std (bool): If True, applies standard normalization before classification.
        norm (callable): Normalization function for feature extraction on the extracted feature.

    Returns:
        list: [train_acc, val_acc] containing training and validation accuracy.
    """
    
    current_rng_state = torch.get_rng_state()
    test = not search
    if test:
        valloader = tsloader
    
    lengths = sum(dims[i] for i in Layer_out)
    torch.manual_seed(42)

    classifier = nn.Sequential(
        nn.Dropout(out_dropout),
        nn.Linear(lengths, 10)
    ).to(device)
    
    if torch.cuda.device_count() > 2:
        classifier = torch.nn.DataParallel(classifier)

    sup_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    sup_lr_scheduler = CustomStepLR(sup_optimizer, nb_epochs=50)
    criterion = nn.CrossEntropyLoss()

    def train_or_evaluate(loader, is_train=True):
        nonlocal classifier
        if is_train:
            classifier.train()
        else:
            classifier.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for x, labels in loader:
            if pre_std:
                 x = stdnorm(x, dims=[1])
            
            x, _ = get_pos_neg_batch_imgcats(x, x, p=1)
            x, labels = x.to(device), labels.to(device)
            
            if is_train:
                classifier.train()
                sup_optimizer.zero_grad()
            
            outputs = process_batch(nets, x, Layer_out, norm)
            outputs = classifier(outputs)
            loss = criterion(outputs, labels)
            
            if is_train:
                loss.backward()
                sup_optimizer.step()
            
            running_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / total

        return accuracy, avg_loss

    for j, net in enumerate(nets):
        net.eval()

    for epoch in range(50):
        train_acc, train_loss = train_or_evaluate(suptrloader, is_train=True)
        sup_lr_scheduler.step()

        if epoch % 20 == 0 or epoch == 49:
            print(f'Epoch [{epoch + 1}/50], Loss: {train_loss:.3f}, Accuracy: {train_acc:.2f}%')

            val_acc, val_loss = train_or_evaluate(valloader, is_train=False)
            print(f'Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.3f}')

    if test:
        test_acc, test_loss = train_or_evaluate(tsloader, is_train=False)
        print(f'Test Accuracy: {test_acc:.2f}%')

    torch.set_rng_state(current_rng_state)

    return [train_acc, val_acc]



def create_layer(layer_config,opt_config, load_params, device):
    layer_num = layer_config['num']-1

    net = Layer(layer_config["ch_in"], layer_config["channels"]
    , bias=True, Norm = "stdnorm",act = layer_config["act"])
    
    if load_params:
        net.load_state_dict(torch.load('./results/params_MNIST_l' + str(layer_num) +'.pth', map_location='cpu'))
        for param in net.parameters():
            param.requires_grad = False

    net.to(device)
    optimizer = AdamW(net.parameters(), lr=opt_config["lr"], weight_decay=opt_config["weight_decay"])
    scheduler = ExponentialLR(optimizer, opt_config["gamma"])

    return net, optimizer, scheduler

def hypersearch(epochs , a, b
    , NL, Layer_out, tr_and_eval 
    , stdnorm_out 
    ,search, device_num, loaders,p, pre_std, seed_num):
    """
    Conducts training and evaluating for a neural network model based on the given hyperparameter configurations.

    Args:
        epochs (int): Number of training epochs.
        a (float): Scaling parameter for positive loss term.
        b (float): Scaling parameter for negative loss term.
        NL (int): Number of layers in the network.
        Layer_out (list): List of layers used for the final classification.
        tr_and_eval (bool): Whether to perform training and evaluation simultaneously.
        stdnorm_out (str): Type of normalization to apply (e.g., "L2norm" or "stdnorm").
        search (bool): If True, uses test set as validation set.
        device_num (int): GPU device number to use for training.
        loaders (tuple): Tuple containing (trainloader, valloader, testloader, suptrloader).
        p (int): Number of negative samples per positive sample.
        pre_std (bool): Whether to apply pre-standardization.
        seed_num (int): Random seed for reproducibility.

    Returns:
        tuple: Contains:
            - tacc (float): Test accuracy.
            - all_pos (list): List of goodness for positive examples per layer per epoch.
            - all_neg (list): List of goodness for negative examples per layer per epoch.
            - nets (list): List of trained network layers.
    """
    #trainloader, _, testloader,_ = get_train(batchsize, augment, Factor)
    trainloader, valloader, testloader, suptrloader = loaders

    #torch.manual_seed(1234)
    torch.manual_seed(seed_num)
    device = 'cuda:' + str(device_num) if torch.cuda.is_available() else 'cpu'
    nets = []; optimizers = []; schedulers= []; threshold1 = []; threshold2 = []; lamda = []; period= []
    #trainouts = []; testouts = []; 
    freezelayer = NL-1

    with open('config.json', 'r') as f:
        config = json.load(f)

    for i, (layer_config, opt_config) in enumerate(zip(config['MNIST']['layer_configs'][:NL], config['MNIST']['opt_configs'][:NL])):
        if i < NL-1:
            load_params = True
        if i == NL-1:
            load_params = False
        net, optimizer, scheduler = create_layer(layer_config, opt_config, load_params = load_params, device=device)
        nets.append(net)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        threshold1.append(opt_config['th1'])
        threshold2.append(opt_config['th2'])
        lamda.append(opt_config['lamda'])
        period.append(opt_config['period'])
    
    for (net, concat) in zip(nets, layer_config['concat']):
        net.concat = concat

    if stdnorm_out == "L2norm":#L2norm
        norm = L2norm(dims = [1])
    elif stdnorm_out == "stdnorm":
        norm = standardnorm(dims = [1])
    else:
        norm = nn.Identity()
    

    if tr_and_eval:
        nets, all_pos, all_neg, _, tacc = train(
        nets, device, optimizers,schedulers, threshold1,threshold2, epochs
            , a,b, lamda, freezelayer,period,tr_and_eval, Layer_out,trainloader
            , valloader, testloader, suptrloader, opt_config['out_dropout'],
            search,p, pre_std, norm, layer_config["aug"])

    else:
        nets, all_pos, all_neg, Dims = train(
            nets, device, optimizers,schedulers, threshold1,threshold2, epochs
            , a,b, lamda, freezelayer,period,tr_and_eval, Layer_out,trainloader
            , valloader, testloader, suptrloader, opt_config['out_dropout'],
            search,p, pre_std, norm,layer_config["aug"])
        
        tacc = evaluation_(nets, Dims, device, Layer_out
            ,opt_config['out_dropout'], search,valloader, testloader, suptrloader, pre_std, norm)
        
    return tacc, all_pos, all_neg, nets

def main(epochs, device_num, tr_and_eval, save_model, loaders, NL):
    """
    Main function to conduct training, evaluation, and model saving.

    Args:
        epochs (int): Number of training epochs.
        device_num (int): Device ID for CUDA (GPU).
        tr_and_eval (bool): Whether to train and evaluate simultaneously.
        save_model (bool): Whether to save the trained model.
        loaders (tuple): Tuple containing (trainloader, valloader, testloader, suptrloader).
        NL (int): Number of layers in the neural network.

    Returns:
        float: Final test accuracy after training and evaluation.
    """

    # Perform hyperparameter search and training
    tacc, all_pos, all_neg, nets = hypersearch(
        epochs=epochs,  # Set the number of epochs
        a=1,  # Loss function parameter, default=1
        b=1,  # Loss function parameter, default=1
        cout=False,  # Whether to use all neurons for classification
        NL=NL,  # Number of layers in the model
        Layer_out=[0,1],  # Output layers used for classification
        tr_and_eval=tr_and_eval,  # Whether to train and evaluate
        pre_std=True,  # Apply standardization before input
        stdnorm_out=True,  # Apply standardization before classification
        search=False,  # If False, uses test set for validation
        device_num=device_num,  # GPU device number
        loaders=loaders,  # Data loaders
        p=1,  # Number of negative samples for contrastive learning
        seed_num=1234  # Set random seed for reproducibility
    )

    # Save model checkpoints if save_model is True
    if save_model:
        for i, net in enumerate(nets):
            model_path = f'./results/params_MNIST_l{i}.pth'
            torch.save(net.state_dict(), model_path)
            print(f"Model layer {i} saved to {model_path}")

    return tacc



if __name__ == "__main__":
  
    # Parse command-line arguments
    parser = argparse.ArgumentParser('SCFF script', parents=[get_arguments()])
    args = parser.parse_args()

    # Print parsed arguments
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")

    # Load dataset and prepare data loaders
    loaders = get_train(batchsize=100, augment=True)

    # Call the main function with parsed arguments
    tsacc = main(
        epochs=args.epochs,
        device_num=args.device_num,
        tr_and_eval=args.tr_and_eval,
        save_model=args.save_model,
        loaders=loaders,
        NL=args.NL
    )
    