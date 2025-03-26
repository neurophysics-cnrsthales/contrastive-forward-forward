import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import ExponentialLR, StepLR, LinearLR

import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt

import torchvision
from torchvision.transforms import transforms, ToPILImage
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split,Subset
from torchvision.transforms import transforms, ToPILImage, Compose, ToTensor,RandomAffine, Normalize, Lambda
from torchvision.datasets import MNIST
import argparse
import time

import numpy as np
from numpy import fft 
import math
import optuna
import json

torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

def get_arguments():
    parser = argparse.ArgumentParser(description="SCFF MNIST Training")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--tr_and_eval", action='store_true')
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--NL", type=int, default=3)
    
    parser.add_argument("--lr1", type=float, default=0.004)
    parser.add_argument("--lr2", type=float, default=0.003)
    parser.add_argument("--lr3", type=float, default=0.0001)
    
    parser.add_argument("--weight_decay", type=float, default=0.0003)
    parser.add_argument("--gamma1", type=float, default=0.7)
    parser.add_argument("--gamma2", type=float, default=0.7)
    parser.add_argument("--gamma3", type=float, default=0.7)
    
    parser.add_argument("--th1", type=int, default=1)
    parser.add_argument("--th2", type=int, default=3)
    parser.add_argument("--th3", type=int, default=6)
    parser.add_argument("--lamda", type=float, default=0.0)
    parser.add_argument("--period", type=int, default=500)
    
    parser.add_argument("--act1", type=str, default="triangle")
    parser.add_argument("--act2", type=str, default="relu")
    parser.add_argument("--act3", type=str, default="triangle")
    parser.add_argument("--out_dropout", type=float, default=0.2)
    parser.add_argument("--seed_num", type=int, default=10)

    parser.add_argument("--concats", type=lambda x: tuple(map(int, x.split(","))),
                        default=(1, 1, 1), help="Comma-separated e.g. 1,1,1")
    parser.add_argument("--layer_out", type=lambda x: tuple(map(int, x.split(","))),
                        default=(1, 2), help="Comma-separated e.g. 1,2")

    return parser


#custom the trainloader to include the augmented views of the original batch
torch.manual_seed(1234)
# Define the two sets of transformations


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
        img = ToPILImage()(img)

        # Apply the original transform
        if self.transform is not None:
            orig_img = self.transform(img)

        return orig_img, target

def get_train(batchsize, augment):
    torch.manual_seed(1234)
    # Transformation pipeline
    transform = Compose([
        ToTensor(),
        #Lambda(lambda x: torch.flatten(x))
        ])

    transform_tr = Compose([
        RandomAffine(degrees=0, translate=(2/28, 2/28)),
        ToTensor(),
        #Lambda(lambda x: torch.flatten(x))
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

    #mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, val_size])

    
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

class Conv2d(nn.Module):
    """
    A custom 2D convolutional layer with optional normalization/standardization, activation, 
    and concatenation of input channels for self-contrastive inputs.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        kernel_size (tuple or int): Size of the convolutional kernel.
        pad (int, optional): Padding size.
        batchnorm (bool, optional): Whether to apply batch normalization. Default is False.
        normdims (list, optional): Dimensions to apply normalization over. Default is [1,2,3].
        norm (str, optional): Normalization type, 'stdnorm' for standard normalization or 'L2norm'. Default is 'stdnorm'.
        bias (bool, optional): Whether to use bias in convolution. Default is True.
        dropout (float, optional): Dropout rate. Default is 0.0.
        padding_mode (str, optional): Padding mode for convolution (e.g., 'zeros', 'reflect'). Default is 'reflect'.
        concat (bool, optional): Whether to split input channels and apply convolution separately before summing. Default is True.
        act (str, optional): Activation function for transmitting information to the next layer not for plastisity, 'relu' or 'triangle'. Default is 'relu'.
    """
    def __init__(
        self, 
        input_channels, 
        output_channels, 
        kernel_size, 
        pad=0, 
        batchnorm=False, 
        normdims=[1,2,3], 
        norm="stdnorm",
        bias=True, 
        dropout=0.0, 
        padding_mode="reflect", 
        concat=True, 
        act="relu"
    ):
        super(Conv2d, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.normdims = normdims
        self.concat = concat  # If True, input channels are split and processed separately because of concatenated pos/neg images
        self.relu = torch.nn.ReLU()

        # Define convolutional layer
        # Weights for first layer: [output_channels, 32, 32, input_channels, kernel_size[0], kernel_size[1]]
        self.conv_layer = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=kernel_size, 
            bias=bias
        )
        
        # Initialize weights using Xavier uniform initialization
        init.xavier_uniform_(self.conv_layer.weight)
        # Set padding parameters
        self.padding_mode = padding_mode
        self.F_padding = (pad, pad, pad, pad)  # Symmetric padding on all sides
        
        
        # Define activation function
        if act == 'relu':
            self.act = torch.nn.ReLU()
        else:
            self.act = triangle()
        
        # Apply batch normalization if enabled
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(self.input_channels, affine=False)
        else:
            self.bn1 = nn.Identity()

        # Select normalization type (Standard Normalization or L2 Normalization)
        if norm == "L2norm":
            self.norm = L2norm(dims = normdims)
        elif norm == "stdnorm":
            self.norm = standardnorm(dims = normdims)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after convolution without activation.
        """
        # batchnorm is false by default
        x = self.bn1(x) 
        x = F.pad(x, self.F_padding, self.padding_mode)
        x = self.norm(x) #stardardization before convolutions

        if self.concat: 
            lenchannel = x.size(1)//2
             # If concat mode is enabled, split channels into two halves, apply convolution separately, and sum results
            out = self.conv_layer(x[:, :lenchannel]) + self.conv_layer(x[:, lenchannel:])
        else:
            out = self.conv_layer(x)
        
        return out



def train(nets, device, optimizers,schedulers, threshold1,threshold2,  dims_in, dims_out, epochs, pool
            , a,b, lamda, freezelayer,period,extra_pool, tr_and_eval, Layer_out, all,trainloader
            , valloader, testloader, suptrloader,pre_std, stdnorm_out, search, p, config):

    """
    Trains neural network layers in a greedy layer-wise manner (previous layers frozon).

    Args:
        nets (list): List of neural network layers.
        device (str): Device to run computations on ('cuda' or 'cpu').
        optimizers (list): List of optimizers for each layer.
        schedulers (list): Learning rate schedulers for each layer.
        threshold1 (list): List of Threshold values for Positive examples. 
        threshold2 (list): List of Threshold values for negative examples.
        dims_in (tuple): Dimensions for input normalization.
        dims_out (tuple): Dimensions for output normalization.
        epochs (int): Number of training epochs.
        pool (list): Pooling layers for each network layer.
        a (float): Scaling parameter for positive sample loss, default=1.
        b (float): Scaling parameter for negative sample loss, default=1.
        lamda (list): Regularization coefficients for each layer.
        freezelayer (int): Number of layers where weights were frozon.
        period (list): Number of batches before updating the learning rate.
        extra_pool (list): Extra pooling layers for feature retrieval.
        tr_and_eval (bool): Whether to evaluate the model during unsupervised SCFF training.
        Layer_out (list): Layers to use for final classifier.
        trainloader (DataLoader): Training dataset loader.
        valloader (DataLoader): Validation dataset loader.
        testloader (DataLoader): Test dataset loader.
        suptrloader (DataLoader): Supervised training dataset loader for evaluation.
        p (int): Number of negative samples per positive sample, default=1. 
        config (dict): Configuration dictionary.

    Returns:
        tuple: (nets, all_pos, all_neg, Dims, taccs) if `tr_and_eval` is True,
               else (nets, all_pos, all_neg, Dims).
    """
    all_pos = [];all_neg= []
    NL = len(nets)
    for i in range(NL):
        all_pos.append([])
        all_neg.append([])
        
    firstpass=True
    nbbatches = 0
    #wsav = []

    NBLEARNINGEPOCHS = epochs

   
    if epochs == 0:
        N_all = NBLEARNINGEPOCHS + 1
    else:
        N_all = NBLEARNINGEPOCHS

    Dims = []
    taccs = []
    # Start the experiment !
    best_acc = 0

    for epoch in range(N_all):

        print("Epoch", epoch)
        #correct=0; total=0
        if epoch < NBLEARNINGEPOCHS and epochs !=0:
            for i, net in enumerate(nets):
                net.train()
            
            print("Unlabeled.")
            UNLAB = True
            zeloader = trainloader
        else: # epoch  < UNLABPERIOD + TRAINPERIOD:
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

        for numbatch, (x, _, _, _) in enumerate(zeloader):
            nbbatches += 1
            x = x.to(device)

            for i in range(NL):
                
                if nets[i].concat:
                    x = stdnorm(x, dims = dims_in)
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)

                x = nets[i](x)
                x_neg = nets[i](x_neg)

                yforgrad = nets[i].relu(x).pow(2).mean([1])
                yforgrad_neg =nets[i].relu(x_neg).pow(2).mean([1])

                if UNLAB :
                    optimizers[i].zero_grad()
                    
                    loss =  torch.log(1 + torch.exp(
                        a*(- yforgrad  + threshold1[i]))).mean([1,2]).mean(
                        ) + torch.log(1 + torch.exp(
                            b*(yforgrad_neg  - threshold1[i]))).mean([1,2]).mean() + lamda[i] * torch.norm(yforgrad, p=2, dim = (1,2)).mean(
                            ) 
                    loss.backward()
                    optimizers[i].step()  

                    if (nbbatches+1)%period[i] == 0:
                        schedulers[i].step()
                        print(f'nbbatches {nbbatches+1} learning rate: {schedulers[i].get_last_lr()[0]}')  
                
                x = pool[i](nets[i].act(x)).detach()
                x_neg = pool[i](nets[i].act(x_neg)).detach()


                if firstpass:
                    print("Layer", i, ": x.shape:", x.shape, "y.shape (after MaxP):", x.shape, end=" ")
                    _, channel, h, w = x.shape
                    Dims.append(channel * h * w)

            firstpass = False
            goodness_pos += (torch.mean(yforgrad.mean([1,2]))).item()
            goodness_neg += (torch.mean(yforgrad_neg.mean([1,2]))).item()

            if UNLAB and numbatch == len(zeloader) - 1:
                print(goodness_pos/len(zeloader), goodness_neg/len(zeloader))
                all_pos[i].append(goodness_pos)
                all_neg[i].append(goodness_neg)
                goodness_pos,  goodness_neg = 0,0           

        if tr_and_eval:
            if (epoch+1)>0 and (epoch+1)%1==0:
                _, tacc = evaluate_model(nets, pool, extra_pool, config, loaders, search, Dims)
                if tacc>best_acc:
                    best_acc = tacc

    print("Training done..")
    
    if tr_and_eval:
        return nets, all_pos, all_neg, Dims, best_acc
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

class EvaluationConfig:
    def __init__(self, device, dims, dims_in, dims_out,stdnorm_out, out_dropout, Layer_out, pre_std, all_neurons):
        self.device = device
        self.dims = dims
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.stdnorm_out = stdnorm_out
        self.out_dropout = out_dropout
        self.Layer_out = Layer_out
        self.all_neurons = all_neurons
        self.pre_std = pre_std

def calculate_output_length(dims, nets, extra_pool, Layer, all_neurons):
    lengths = 0
    if all_neurons:
        for i, length in enumerate(dims):
            if i in Layer:
                lengths += length
    else:
        for i, length in enumerate(dims):
            #print(length)
            if i in Layer:
                len_after_pool = math.ceil((math.sqrt(length / nets[i].output_channels) - extra_pool[i].kernel_size) / extra_pool[i].stride + 1)
                lengths += len_after_pool*len_after_pool * nets[i].output_channels

    return lengths

def build_classifier(lengths, config):
    classifier = nn.Sequential(
        nn.Dropout(config.out_dropout),
        nn.Linear(lengths, 10)  # Assuming output dimension of 10
    ).to(config.device)
    if torch.cuda.device_count() > 2:
        classifier = nn.DataParallel(classifier)
    return classifier

def train_readout(classifier, nets, pool, extra_pool, loader, criterion, optimizer, config, epoch):
    # Training loop implementation
    classifier.train()
    correct = 0
    total = 0

    for i, (x, labels) in enumerate(loader):

        x = x.to(config.device)
        labels = labels.to(config.device)

        outputs = []
        
        with torch.no_grad():
            for j, net in enumerate(nets):
                if net.concat:
                    x = stdnorm(x, dims = config.dims_in)
                    x = torch.cat((x, x), dim=1)

                x = pool[j](net.act(net(x)))

                if not config.all_neurons:
                    out = extra_pool[j](x)

                if config.stdnorm_out:
                    out = stdnorm(out, dims = config.dims_out)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

        outputs = torch.cat(outputs, dim = 1)    
        optimizer.zero_grad()
        outputs = classifier(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

def test_readout(classifier, nets, pool, extra_pool, loader, criterion, config, epoch, mode):
    
    classifier.eval()
    running_loss = 0.
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, (x, labels) in enumerate(loader):
       
            x = x.to(config.device)
            labels = labels.to(config.device)
            outputs = []
            for j, net in enumerate(nets):
                if net.concat:
                    x = stdnorm(x, dims = config.dims_in)
                    x = torch.cat((x, x), dim=1)
                    
                x = pool[j](net.act(net(x)))

                if not config.all_neurons:
                    out = extra_pool[j](x)

                if config.stdnorm_out:
                    out = stdnorm(out, dims = config.dims_out)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

            outputs = torch.cat(outputs, dim = 1) 
            outputs = classifier(outputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    if mode == 'Val':
        print(f'Accuracy of the network on the 10000 '+ mode+ f' images: {100 * correct / total} %')
        print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

    return correct / total

def evaluate_model(nets, pool, extra_pool, config, loaders, search, Dims):
    """
    Evaluates a trained neural network model by training a classifier on top of extracted features.

    Args:
        nets (list): List of trained neural network layers.
        pool (list): List of pooling layers corresponding to each network layer.
        extra_pool (list): Additional pooling layers for feature extraction.
        config (EvaluationConfig): Configuration containing evaluation parameters.
        loaders (tuple): Tuple containing (trainloader, valloader, testloader, suptrloader).
        search (bool): If True, validation is done using the test dataset.
        Dims (list): Dimensions of each layer's output feature map, calculated from calculate_output_length

    Returns:
        tuple: (acc_train, acc_val) where:
            - acc_train (float): Final accuracy on the training dataset.
            - acc_val (float): Final accuracy on the validation dataset.
    """
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    lengths = calculate_output_length(Dims, nets, extra_pool, config.Layer_out, config.all_neurons)
    print(lengths)
    classifier = build_classifier(lengths, config)
    
    _, valloader, testloader, suptrloader = loaders
    # Optimizer and criterion setup
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    lr_scheduler = CustomStepLR(optimizer, nb_epochs=50)
    criterion = nn.CrossEntropyLoss()

    if not search:
        valloader = testloader
    # Main evaluation loop
    for j, net in enumerate(nets):
        net.eval()

    for epoch in range(50):
        acc_train = train_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, optimizer, config, epoch)
        lr_scheduler.step()
        if epoch % 20 == 0 or epoch == 49:
            print(f'Accuracy of the network on the 50000 train images: {100 * acc_train} %')
            acc_train = test_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, config,epoch, 'Train')
            acc_val = test_readout(classifier, nets, pool, extra_pool, valloader, criterion, config,epoch, 'Val')

    torch.set_rng_state(current_rng_state)
    
    return acc_train, acc_val

def create_layer(layer_config,opt_config, load_params, device, act):
    layer_num = layer_config['num']-1

    net = Conv2d(layer_config["ch_in"], layer_config["channels"], (layer_config["kernel_size"], layer_config["kernel_size"]),
                  pad = layer_config["pad"], norm = "stdnorm", padding_mode = layer_config["padding_mode"], act = act)
    
    if load_params:
        net.load_state_dict(torch.load('./results/params_MNIST_CNN_l' + str(layer_num) +'.pth', map_location='cpu'))
        for param in net.parameters():
            param.requires_grad = False

    if layer_config["pooltype"] == 'Avg':
        pool = nn.AvgPool2d(kernel_size=layer_config["pool_size"], stride=layer_config["stride_size"], padding=layer_config["padding"], ceil_mode=True)
    else:
        pool = nn.MaxPool2d(kernel_size=layer_config["pool_size"], stride=layer_config["stride_size"], padding=layer_config["padding"], ceil_mode=True)
    
    extra_pool = nn.AvgPool2d(kernel_size= layer_config["extra_pool_size"], stride=layer_config["extra_pool_size"], padding=0, ceil_mode=True)
    net.to(device)
    optimizer = AdamW(net.parameters(), lr=opt_config["lr"], weight_decay=opt_config["weight_decay"])
    scheduler = ExponentialLR(optimizer, opt_config["gamma"])

    return net, pool, extra_pool, optimizer, scheduler


def hypersearch(dims, dims_in, dims_out, Batchnorm, epochs
    , a, b, all_neurons, NL, Layer_out, tr_and_eval
    ,pre_std, stdnorm_out, search, device_num, loaders,p,seed_num
    ,lr,weight_decay,gamma,threshold1,threshold2,lamda,period,act,out_dropout,concats):
    """
    Conducts layer-wise training of a multi-layer SCFF network with separate hyperparameters per layer.

    Args:
        dims, dims_in, dims_out: Dimensionality settings for standardization.
        Batchnorm: Not used here but passed for potential compatibility.
        epochs: Max training epochs.
        a, b: Loss function parameters for positive and negative samples, default=1, not used.
        all_neurons: Flag for using all neurons in readout.
        NL: Number of layers.
        Layer_out: List of layer indices used for evaluation.
        tr_and_eval: Whether to perform evaluation during training.
        pre_std, stdnorm_out: Standardization control flags.
        search: Flag indicating if in hyperparameter search mode.
        device_num: CUDA device number.
        loaders: Tuple of train/val/test/supervised loaders.
        p: Number of negative samples.
        seed_num: Random seed for reproducibility.
        lr, weight_decay, gamma: Lists of optimizer parameters per layer.
        threshold1, threshold2: Goodness thresholds for pos/neg samples per layer.
        lamda: L2 norm regularization per layer.
        period: Learning rate scheduler step frequency per layer.
        act: Activation function per layer.
        out_dropout: Dropout before classifier.
        concats: Tuple of boolean flags per layer indicating channel concatenation.

    Returns:
        tacc: Evaluation accuracy
        all_pos, all_neg: Per-layer average goodness scores
        nets: List of trained networks
    """
    trainloader, valloader, testloader, suptrloader = loaders

    torch.manual_seed(seed_num)
    
    device = 'cuda:' + str(device_num) if torch.cuda.is_available() else 'cpu'
    nets = []; optimizers = []; schedulers= []#; threshold1 = []; threshold2 = []; lamda = []; period= []
    pools = []; extra_pools = []

    with open('config.json', 'r') as f:
        config = json.load(f)

    freezelayer = 0
    
    for i, (layer_config, opt_config) in enumerate(zip(config['MNIST_CNN']['layer_configs'][:NL], config['CIFAR']['opt_configs'][:NL])):
        
        load_params = False
        net, pool, extra_pool, _, _ = create_layer(layer_config, opt_config
                                                                   , load_params = load_params, device=device, act=act[i])
        nets.append(net)
        pools.append(pool)
        extra_pools.append(extra_pool)
        optimizer = AdamW(net.parameters(), lr=lr[i], weight_decay=weight_decay[i])
        optimizers.append(optimizer)
        schedulers.append(ExponentialLR(optimizer, gamma[i]))
        
    
    for (net, concat) in zip(nets, concats):
        net.concat = concat
        
    config = EvaluationConfig(device=device, dims=dims, dims_in=dims_in, dims_out=dims_out, stdnorm_out = stdnorm_out, 
                              out_dropout=out_dropout, Layer_out=Layer_out,pre_std = pre_std, all_neurons = all_neurons)

    if tr_and_eval:
        nets, all_pos, all_neg, _, tacc = train(
        nets, device, optimizers,schedulers, threshold1,threshold2, dims_in, dims_out, epochs, pools, a,b, lamda, freezelayer
        ,period, extra_pools, tr_and_eval, Layer_out, all_neurons,trainloader, valloader, testloader, suptrloader,pre_std, stdnorm_out
        ,search, p, config)

    else:
        nets, all_pos, all_neg, Dims = train(
            nets, device, optimizers,schedulers, threshold1,threshold2, dims_in, dims_out, epochs, pools, a,b, lamda, freezelayer
            ,period, extra_pools, tr_and_eval, Layer_out, all_neurons,trainloader, valloader, testloader, suptrloader,pre_std, stdnorm_out
            ,search, p, config)
        _, tacc = evaluate_model(nets, pools, extra_pools, config, loaders, search, Dims) 
        
    return tacc, all_pos, all_neg, nets


def main(epochs, device_num, tr_and_eval, save_model, loaders, NL,
         lr1, lr2, lr3, weight_decay, gamma1, gamma2, gamma3,
         th1, th2, th3, lamda, period, act1, act2, act3,
         out_dropout, Layer_out, seed_num, concats):
    """
    Entry point for training and evaluation of a 3-layer SCFF-based CNN on MNIST.

    Args:
        epochs: Total training epochs.
        device_num: CUDA device ID.
        tr_and_eval: Whether to evaluate during training.
        save_model: Flag to save final weights to disk.
        loaders: Dataloaders for train, val, test, and supervised training.
        NL: Number of layers.
        lr1, lr2, lr3: Learning rates for each layer.
        weight_decay: Shared weight decay for all layers.
        gamma1-3: LR decay rates.
        th1-3: Thresholds for positive/negative loss.
        lamda: L2 regularization.
        period: Scheduler step frequency.
        act1-3: Activations for each layer.
        out_dropout: Dropout for classifier head.
        Layer_out: Tuple of which layers to use for final evaluation.
        seed_num: Random seed.
        concats: Which layers use feature concatenation.

    Returns:
        Final test accuracy.
    """
    tacc, all_pos, all_neg, nets = hypersearch(
        dims=(1, 2, 3),
        dims_in=(1, 2, 3),
        dims_out=(1, 2, 3),
        Batchnorm=False,
        epochs=epochs,
        a=1,
        b=1,
        all_neurons=False,
        NL=NL,
        Layer_out=Layer_out,
        tr_and_eval=tr_and_eval,
        pre_std=True,
        stdnorm_out=True,
        search=False,
        device_num=device_num,
        loaders=loaders,
        p=1,
        seed_num=seed_num,
        lr=[lr1, lr2, lr3],
        weight_decay=[weight_decay] * 3,
        gamma=[gamma1, gamma2, gamma3],
        threshold1=[th1, th2, th3],
        threshold2=[th1, th2, th3],
        lamda=[lamda] * 3,
        period=[period] * 3,
        act=[act1, act2, act3],
        out_dropout=out_dropout,
        concats=concats
    )

    if save_model:
        for i, net in enumerate(nets):
            torch.save(net.state_dict(), f'./results/params_MNIST_CNN_l{i}.pth')

    return tacc


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()

    print("Running with the following arguments:")
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")

    loaders = get_train(batchsize=100, augment=True)

    tsacc = main(
        epochs=args.epochs,
        device_num=args.device_num,
        tr_and_eval=args.tr_and_eval,
        save_model=args.save_model,
        loaders=loaders,
        NL=args.NL,
        lr1=args.lr1,
        lr2=args.lr2,
        lr3=args.lr3,
        weight_decay=args.weight_decay,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        gamma3=args.gamma3,
        th1=args.th1,
        th2=args.th2,
        th3=args.th3,
        lamda=args.lamda,
        period=args.period,
        act1=args.act1,
        act2=args.act2,
        act3=args.act3,
        out_dropout=args.out_dropout,
        Layer_out=args.layer_out,
        seed_num=args.seed_num,
        concats=args.concats
    )

