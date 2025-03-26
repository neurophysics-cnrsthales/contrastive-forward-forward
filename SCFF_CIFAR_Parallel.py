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
import argparse
import time

import numpy as np
from numpy import fft 
import math
import optuna
import json


#custom the trainloader to include the augmented views of the original batch
torch.manual_seed(1234)
# Define the two sets of transformations
#BATCHSIZE = 50
s = 0.5
transform1 = transforms.Compose([
    #transforms.RandomCrop(32, padding=0),
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # using default scale range
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform2 = transforms.Compose([
    #transforms.RandomCrop(32, padding=0),
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # using default scale range
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding = 1),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#transform_test = transform_train

transform_test = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class DualAugmentCIFAR10(torchvision.datasets.CIFAR10):
    """
    Custom CIFAR-10 dataset that applies dual augmentation techniques 
    for unsupervised SCFF. (default: no augmentation is used) 

    Args:
        root (str): Root directory where the dataset is stored.
        augment (str): Type of augmentation to apply. 
                       Options: 'no' (default), 'single', 'dual'.
        *args: Additional arguments for the CIFAR-10 dataset.

    Attributes:
        augment (str): Stores the selected augmentation mode.
    """
    def __init__(self, root, augment="No", *args, **kwargs):
        super(DualAugmentCIFAR10, self).__init__(root,*args, **kwargs)
        self.augment = augment
        
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img_pil = ToPILImage()(img)
        img_original = transform_train(img_pil)

        if self.augment == "single":
            img1 = transform1(img_pil)
            return img_original, img1, img_original, target
        elif self.augment == "dual":
            img1 = transform1(img_pil)
            img2 = transform2(img_pil)
            return img_original, img1, img2, target
        else:
            return img_original, target

            
class DualAugmentCIFAR10_test(torchvision.datasets.CIFAR10):
    """
    Custom CIFAR-10 dataset that applies augmentation techniques 
    for supervised evaluation of the trained model with SCFF.

    Args:
        aug (bool): Whether to apply data augmentation to test images.
        *args: Additional arguments for the CIFAR-10 dataset.

    Attributes:
        aug (bool): Stores whether augmentation is applied. True for train set, False for test set
    """
    def __init__(self, aug=False, *args, **kwargs):
        super(DualAugmentCIFAR10_test, self).__init__(*args, **kwargs)
        self.aug = aug
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = ToPILImage()(img)
        
        if self.aug:
            img = transform_train(img)
        else:
            img = transform_test(img)
        
        return img, target

# Define the custom CIFAR-10 dataset

def get_train(batchsize, augment, Factor):
    """
    Creates data loaders for CIFAR-10 training and validation.

    Args:
        batchsize (int): Batch size for training.
        augment (str): Data augmentation strategy (e.g., 'no', 'single', 'dual').
        factor (float): Proportion of dataset to use for training.

    Returns:
        tuple: (train_loader, val_loader, test_loader, sup_train_loader)
    """
    torch.manual_seed(1234)
    trainset = DualAugmentCIFAR10(root='./data', train=True, download=True, augment=augment)
    sup_trainset = DualAugmentCIFAR10_test(root='./data', aug = True, train=True, download=True)
    # Create a DataLoader
    factor = Factor
    train_len = int(len(trainset) * factor)
    #val_len = len(trainset) - train_len

    indices = torch.randperm(len(trainset)).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    # Create subsets
    train_data = Subset(trainset, train_indices)
    sup_train_data = Subset(sup_trainset, train_indices)
    val_data = Subset(sup_trainset, val_indices)

    testset = DualAugmentCIFAR10_test(root='./data',aug = False, train=False, download=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    #train_data, val_data = random_split(trainset, [train_len, val_len])

    trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)

    if factor ==1:
        valloader = testloader
    else:
        valloader = DataLoader(val_data, batch_size=1000, shuffle=True, num_workers=2)

    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True, )

    return trainloader, valloader, testloader, sup_trainloader

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
# with padding version
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
            , valloader, testloader, suptrloader,pre_std, stdnorm_out, search, p
            , config, alleps):
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
        alleps (list): Epochs per layer.

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
    
    NBLEARNINGEPOCHS = epochs

    if epochs == 0:
        N_all = NBLEARNINGEPOCHS +1
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

        for numbatch, (x, _) in enumerate(zeloader):
            #print(numbatch)
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

                if i < freezelayer:
                    UNLAB = False
                    
                else:
                    UNLAB = True

                if UNLAB and epoch<alleps[i]:
                    optimizers[i].zero_grad()
                    loss =  torch.log(1 + torch.exp(
                        a*(- yforgrad  + threshold1[i]))).mean([1,2]).mean(
                        ) + torch.log(1 + torch.exp(
                            b*(yforgrad_neg  - threshold2[i]))).mean([1,2]).mean() + lamda[i] * torch.norm(yforgrad, p=2, dim = (1,2)).mean(
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
            if epoch>10 and (epoch+1)%1==0:
                _, tacc = evaluate_model(nets, pool, extra_pool, config, loaders, search, Dims)
                if tacc > best_acc:
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
        #threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
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
        net.load_state_dict(torch.load('./results/params_CIFAR_l' + str(layer_num) +'.pth', map_location='cpu'))
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
    ,lr,weight_decay,gamma,threshold1,threshold2,lamda,period,out_dropout,act, concats, alleps):
    """
    Conducts a unsupervised SCFF training followed by supervised evaluation of the trained networks.
    Args:
        dims (tuple): Input feature dimensions.
        dims_in (tuple): Input normalization dimensions.
        dims_out (tuple): Output normalization dimensions.
        Batchnorm (bool): Whether to use batch normalization.
        epochs (int): Number of epochs to train.
        a (float): Contrastive loss parameter.
        b (float): Contrastive loss parameter.
        all_neurons (bool): If True, uses all neurons.
        NL (int): Number of layers in the network.
        Layer_out (list): Indices of layers used for output.
        tr_and_eval (bool): If True, perform evaluation while training.
        pre_std (bool): If True, standardize inputs before training.
        stdnorm_out (bool): If True, standardize outputs.
        search (bool): If True, enables parameter search mode.
        device_num (int): GPU device number.
        loaders (tuple): Data loaders for training, validation, and testing.
        p (int): Number of negative samples per positive sample.
        seed_num (int): Random seed for reproducibility.
        lr (list): Learning rates for each layer.
        weight_decay (list): Weight decay values per layer.
        gamma (list): Learning rate decay factors.
        threshold1 (list): Positive sample contrastive thresholds.
        threshold2 (list): Negative sample contrastive thresholds.
        lamda (list): Regularization coefficients per layer.
        period (list): Scheduler update periods per layer.
        out_dropout (float): Dropout rate for output layers.
        act (list): Activation functions for each layer.
        concats (tuple): Whether layers should use concatenation.
        alleps (list): Number of epochs per layer.

    Returns:
        tuple: Accuracy, positive samples log, negative samples log, test outputs, trained networks.
    """

    trainloader, valloader, testloader, suptrloader = loaders
    torch.manual_seed(seed_num)
    
    device = 'cuda:' + str(device_num) if torch.cuda.is_available() else 'cpu'
    nets = []; optimizers = []; schedulers= []#; threshold1 = []; threshold2 = []; lamda = []; period= []
    pools = []; extra_pools = []

    with open('config.json', 'r') as f:
        config = json.load(f)

    freezelayer = 0
    
    for i, (layer_config, opt_config) in enumerate(zip(config['CIFAR']['layer_configs'][:NL], config['CIFAR']['opt_configs'][:NL])):
        
        load_params = False
        net, pool, extra_pool, _, _ = create_layer(layer_config, opt_config
                                                                   , load_params = load_params, device=device, act = act[i])
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
        ,search, p, config, alleps)

    else:
        nets, all_pos, all_neg, Dims = train(
            nets, device, optimizers,schedulers, threshold1,threshold2, dims_in, dims_out, epochs, pools, a,b, lamda, freezelayer
            ,period, extra_pools, tr_and_eval, Layer_out, all_neurons,trainloader, valloader, testloader, suptrloader,pre_std, stdnorm_out
            ,search, p, config, alleps)
        _, tacc = evaluate_model(nets, pools, extra_pools, config, loaders, search, Dims) 
        
    return tacc, all_pos, all_neg, nets

def main(device_num,tr_and_eval 
         ,save_model, loaders, NL, lr, weight_decay
         , gamma, lamda, threshold1,threshold2, act, concats,period, alleps, seed_num):
    """
    Main function to run training and evaluation.
    
    Args:
        device_num (int): GPU device index.
        tr_and_eval (bool): Whether to train and evaluate.
        save_model (bool): Whether to save the trained model.
        loaders (tuple): Data loaders for training and evaluation.
        NL (int): Number of layers.
        lr (list): Learning rates per layer.
        weight_decay (list): Weight decay values per layer.
        gamma (list): Learning rate decay factors.
        lamda (list): Regularization lambda per layer.
        threshold1 (list): SCFF loss threshold for positive samples.
        threshold2 (list): SCFF loss threshold for negative samples.
        act (list): Activation functions per layer.
        concats (tuple): Concatenation settings per layer.
        period (list): Scheduler update periods per layer.
        alleps (list): Epochs per layer.
        seed_num (int): Random seed.

    Returns:
        float: Final training accuracy.
    """

    tacc, all_pos, all_neg, nets = hypersearch(
        dims =  (1,2,3),
        dims_in = (1,2,3), 
        dims_out = (1,2,3),
        Batchnorm = False, 
        epochs = max(alleps), 
        a = 1,
        b = 1,
        all_neurons = False,
        NL = NL,
        Layer_out = [2,1,0],
        tr_and_eval = tr_and_eval,
        pre_std = True,
        stdnorm_out = True,
        search = False,
        device_num = device_num,
        loaders = loaders,
        p = 1, 
        seed_num = seed_num,
        lr = lr,
        weight_decay = weight_decay,
        gamma = gamma,
        threshold1 = threshold1,
        threshold2 = threshold2,
        lamda = lamda,
        period = period,
        out_dropout = 0.2,
        act = act,
        concats = concats,
        alleps = alleps
        )

    # Save trained model if required
    if save_model:
        for i, net in enumerate(nets):
            torch.save(net.state_dict(), f'./results/params_CIFAR_parallel_l{i}.pth')
    return tacc


def get_arguments():
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Contrastive Forward-Forward Training Script", add_help=False)

    parser.add_argument("--lr", nargs='+', type=float, default=[0.02, 0.001, 0.0004], help="Learning rate per layer")
    parser.add_argument("--gamma", nargs='+', type=float, default=[0.99, 0.9, 0.99], help="LR decay rate per layer")
    parser.add_argument("--period", nargs='+', type=int, default=[500, 500, 500], help="LR decay rate period")
    parser.add_argument("--weight_decay", nargs='+', type=float, default=[0.0001, 0.0003, 0.0001], help="Weight decay")
    parser.add_argument("--lamda", nargs='+', type=float, default=[0.0008, 0.0004, 0.0016], help="Regularization lambda")
    
    # Threshold values for contrastive loss
    parser.add_argument("--th1", nargs='+', type=int, default=[1, 4, 5], help="Positive sample thresholds")
    parser.add_argument("--th2", nargs='+', type=int, default=[2, 5, 7], help="Negative sample thresholds")

    # Model structure and training
    parser.add_argument("--NL", type=int, default=3, help="Number of layers")
    parser.add_argument("--concats", type=tuple, default=(1, 0, 1), help="Concatenation setting per layer")
    parser.add_argument("--act", nargs='+', type=str, default=["triangle", "triangle", "relu"], help="Activation per layer")
    parser.add_argument("--alleps", nargs='+', type=int, default=[6, 6, 13], help="Epochs per layer")
    
    # Device settings
    parser.add_argument("--device_num", type=int, default=0, help="GPU device to use for training/testing")
    parser.add_argument("--seed_num", type=int, default=1234, help="Random seed for reproducibility")
    
    # Training options
    parser.add_argument("--tr_and_eval", action="store_true", help="Enable training with evaluation")
    parser.add_argument("--save_model", action="store_true", help="Save trained model")
    
    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('ContrastFF script', parents=[get_arguments()])
    args = parser.parse_args()

    # Print argument values
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")

    # Load dataset
    loaders = get_train(batchsize=100, augment="no", Factor=1)

    # Run training
    tsacc = main(
        device_num=args.device_num,
        tr_and_eval=args.tr_and_eval,
        save_model=args.save_model,
        loaders=loaders,
        NL=args.NL,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        lamda=args.lamda,
        threshold1=args.th1,
        threshold2=args.th2,
        act=args.act,
        concats=args.concats,
        period= args.period,
        alleps=args.alleps,
        seed_num=args.seed_num
    )