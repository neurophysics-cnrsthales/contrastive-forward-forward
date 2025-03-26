import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import ExponentialLR, StepLR, LinearLR
from torchvision.datasets import STL10

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
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

transform2 = transforms.Compose([
    #transforms.RandomCrop(32, padding=0),
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # using default scale range
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

transform_train = transforms.Compose([
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])

transform_test = transforms.Compose([
                #transforms.RandomCrop(96, padding=4),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])

class DualAugmentSTL10(STL10):
    """
    Custom STL-10 dataset that applies dual augmentation techniques 
    for unsupervised SCFF. (default: no augmentation is used) 

    Args:
        root (str): Root directory where the dataset is stored.
        augment (str): Type of augmentation to apply. 
                       Options: 'no' (default), 'single', 'dual'.
        *args: Additional arguments for the STL-10 dataset.

    Attributes:
        augment (str): Stores the selected augmentation mode.
    """
    def __init__(self, root, augment="No", *args, **kwargs):
        super(DualAugmentSTL10, self).__init__(root,*args, **kwargs)
        self.augment = augment
        
    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]
        
        #img = torch.tensor(img, dtype=torch.float)
        img_pil = ToPILImage()(img.transpose(1, 2, 0))
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

            
class STL10_test(STL10):
    """
    Custom STL-10 dataset that applies augmentation techniques 
    for supervised evaluation of the trained model with SCFF.

    Args:
        aug (bool): Whether to apply data augmentation to test images.
    
    Attributes:
        aug (bool): Stores whether augmentation is applied. True for train set, False for test set
    """
    def __init__(self, aug=False, *args, **kwargs):
        super(STL10_test, self).__init__(*args, **kwargs)
        self.aug = aug
        
    def __getitem__(self, index):
        
        img, target = self.data[index], self.labels[index]
        # Convert the image to PIL format
        img = ToPILImage()(img.transpose(1, 2, 0))
        
        if self.aug:
            img = transform_train(img)
        else:
            img = transform_test(img)
        
        return img, target

# Define the custom CIFAR-10 dataset

def get_train(batchsize, augment, Factor):

    torch.manual_seed(1234)
    """
    # split can be: 
    'train': The labeled training set.
    'test': The labeled test set.
    'unlabeled': The unlabeled dataset which can be used for unsupervised learning.
    'train+unlabeled': Combines both the labeled training set and the unlabeled set, which can be useful for semi-supervised learning approaches.
    """
    num_workers = 4
    trainset = DualAugmentSTL10(root='./data', split = 'unlabeled', download=True, augment=augment) # split: 
    sup_trainset = STL10_test(root='./data', aug = True, split = 'train', download=True)
    # Create a DataLoader
    factor = Factor
    sup_train_len = int(len(sup_trainset) * factor)
    #val_len = len(trainset) - train_len

    indices = torch.randperm(len(sup_trainset)).tolist()
    sup_train_indices = indices[:sup_train_len]
    sup_val_indices = indices[sup_train_len:]

    # Create subsets
    train_data = trainset

    sup_train_data = Subset(sup_trainset, sup_train_indices)
    val_data = Subset(sup_trainset, sup_val_indices)

    testset = STL10_test(root='./data',aug = False, split = 'test', download=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    #train_data, val_data = random_split(trainset, [train_len, val_len])

    trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=num_workers)

    if factor <1:
        print("using the "+ str(factor*100) +"% train data")
        valloader = DataLoader(val_data, batch_size=100, shuffle=True, num_workers=num_workers)
    else:
        print("using the whole train data")
        valloader = testloader

    #testset = DualAugmentCIFAR10_test(root='./data', train=False, download=True)
    #testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    #sup_trainset = DualAugmentCIFAR10_test(root='./data', train=True, download=True)
    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True, num_workers=num_workers)

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
            , a,b, lamda, freezelayer,period,extra_pool, tr_and_eval, Layer_out, trainloader
            , valloader, testloader, suptrloader,  p, config):
    """
    SCFF train neural network layers in a greedy layer-wise manner (previous layers frozon).

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
        if epoch < NBLEARNINGEPOCHS and epochs !=0:
            nets[-1].train()
            print("Unlabeled.")
            UNLAB = True; 
            zeloader = trainloader
        else: # # evaluate the output neurons without train (freezelayer = NL)
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

        for numbatch, (x, _) in enumerate(zeloader):

            nbbatches += 1
            x = x.to(device)

            for i in range(NL):
                
                if nets[i].concat:
                    x = stdnorm(x, dims = dims_in)
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)

                x = nets[i](x)
                x_neg = nets[i](x_neg)

                yforgrad = nets[i].relu(x).pow(2).mean([1])
                yforgrad_neg = nets[i].relu(x_neg).pow(2).mean([1])

                if i < freezelayer:
                    UNLAB = False
                else:
                    UNLAB = True

                if UNLAB :
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
            if epoch>0 and epoch%1==0:
                tacc = evaluate_model(nets, pool, extra_pool, config, loaders, search, Dims)
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
        print(f'Accuracy of the network on the '+ mode+ f' images: {100 * correct / total} %')
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

    epochs = 100
    lr_scheduler = CustomStepLR(optimizer, nb_epochs=epochs)
    criterion = nn.CrossEntropyLoss()

    if not search:
        valloader = testloader
    # Main evaluation loop
    for j, net in enumerate(nets):
        net.eval()

    for epoch in range(epochs):
        acc_train = train_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, optimizer, config, epoch)
        lr_scheduler.step()
        if epoch % 20 == 0 or epoch == 49:
            print(f'Accuracy of the network on the 100000 train images: {100 * acc_train} %')
            acc_train = test_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, config,epoch, 'Train')
            acc_val = test_readout(classifier, nets, pool, extra_pool, valloader, criterion, config,epoch, 'Val')

    torch.set_rng_state(current_rng_state)
    
    return acc_train, acc_val



def create_layer(layer_config,opt_config, load_params, device):
    layer_num = layer_config['num']-1

    net = Conv2d(layer_config["ch_in"], layer_config["channels"], (layer_config["kernel_size"], layer_config["kernel_size"]),
                  pad = layer_config["pad"], norm = "stdnorm", padding_mode = layer_config["padding_mode"], act = layer_config["act"])
    
    if load_params:
        net.load_state_dict(torch.load('./results/params_STL_l' + str(layer_num) +'.pth', map_location='cpu'))
        for param in net.parameters():
            param.requires_grad = False

    if layer_config["pooltype"] == 'Avg':
        pool = nn.AvgPool2d(kernel_size=layer_config["pool_size"], stride=layer_config["stride_size"], padding=layer_config["padding"], ceil_mode=True)
    else:
        pool = nn.MaxPool2d(kernel_size=layer_config["pool_size"], stride=layer_config["stride_size"], padding=layer_config["padding"], ceil_mode=True)

    if layer_config["extra_pooltype"] == 'Avg':
        extra_pool = nn.AvgPool2d(kernel_size= layer_config["extra_pool_size"], stride=layer_config["extra_pool_size"], padding=0, ceil_mode=True)
    else:
        extra_pool = nn.MaxPool2d(kernel_size= layer_config["extra_pool_size"], stride=layer_config["extra_pool_size"], padding=0, ceil_mode=True)
        
    net.to(device)
    optimizer = AdamW(net.parameters(), lr=opt_config["lr"], weight_decay=opt_config["weight_decay"])
    scheduler = ExponentialLR(optimizer, opt_config["gamma"])

    return net, pool, extra_pool, optimizer, scheduler

def hypersearch(dims, dims_in, dims_out, Batchnorm, epochs
    , a, b, all_neurons, NL, Layer_out, tr_and_eval
    ,pre_std, stdnorm_out, search, device_num, loaders,p,seed_num):

    """
    Conducts training and evaluating for a neural network model based on the given hyperparameter configurations.

    Args:
        dims (tuple): Dimensions for normalization.
        dims_in (tuple): Input dimensions for standardization.
        dims_out (tuple): Output dimensions for standardization.
        Batchnorm (bool): Whether to use batch normalization.
        epochs (int): Number of training epochs.
        a (float): Parameter for loss function, default=1.
        b (float): Parameter for loss function, default=1.
        all_neurons (bool): Whether to use all neurons for classification.
        NL (int): Number of layers in the network.
        Layer_out (list): Indices of layers to use for final classification.
        tr_and_eval (bool): Whether to train and evaluate simultaneously.
        pre_std (bool): Whether to apply pre-standardization (not used).
        stdnorm_out (bool): Whether to apply standardization before classification.
        search (bool): If True, uses test set as validation set.
        device_num (int): GPU device number to use.
        loaders (tuple): Data loaders (train, validation, test, supervised train for final classification).
        p (int): Number of negative samples for each positive.
        seed_num (int): Random seed for reproducibility.

    Returns:
        tuple: (tacc, all_pos, all_neg, testouts, nets) where:
            - tacc (float): Final test/validation accuracy.
            - all_pos (list): List of goodness for positive examples per layer per epoch.
            - all_neg (list): List of goodness for negative examples per layer per epoch.
            - nets (list): Trained network layers.
    """

    trainloader, valloader, testloader, suptrloader = loaders

    torch.manual_seed(seed_num)
    
    device = 'cuda:' + str(device_num) if torch.cuda.is_available() else 'cpu'
    nets = []; optimizers = []; schedulers= []; threshold1 = []; threshold2 = []; lamda = []; period= []
    trainouts = []; testouts = []; 
    pools = []; extra_pools = []

    with open('config.json', 'r') as f:
        config = json.load(f)

    freezelayer = NL-1
    
    for i, (layer_config, opt_config) in enumerate(zip(config['CIFAR']['layer_configs'][:NL], config['CIFAR']['opt_configs'][:NL])):
        if i < NL-1:
            load_params = True
        if i == NL-1:
            load_params = False
        net, pool, extra_pool, optimizer, scheduler = create_layer(layer_config, opt_config
                                                                   , load_params = load_params, device=device)
        nets.append(net)
        pools.append(pool)
        extra_pools.append(extra_pool)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        threshold1.append(opt_config['th1'])
        threshold2.append(opt_config['th2'])
        lamda.append(opt_config['lamda'])
        period.append(opt_config['period'])
    
    for (net, concat) in zip(nets, layer_config['concat']):
        net.concat = concat
        
    config = EvaluationConfig(device=device, dims=dims, dims_in=dims_in, dims_out=dims_out, stdnorm_out = stdnorm_out, 
                              out_dropout=opt_config['out_dropout'], Layer_out=Layer_out,pre_std = pre_std, all_neurons = all_neurons)
    
    if tr_and_eval:
        nets, all_pos, all_neg, _, tacc = train(
        nets, device, optimizers,schedulers, threshold1,threshold2, dims_in, dims_out, epochs, pools, a,b, lamda, freezelayer
        ,period, extra_pools, tr_and_eval, Layer_out, trainloader, valloader, testloader, suptrloader
        ,p, config)

    else:
        nets, all_pos, all_neg, Dims = train(
            nets, device, optimizers,schedulers, threshold1,threshold2, dims_in, dims_out, epochs, pools, a,b, lamda, freezelayer
            ,period, extra_pools, tr_and_eval, Layer_out, trainloader, valloader, testloader, suptrloader
            ,p, config)
        
        tacc = evaluate_model(nets, pools, extra_pools, config, loaders, search, Dims) 
        

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
        dims=(1,2,3),  # Normalization dimensions
        dims_in=(1,2,3),  # Input normalization dimensions
        dims_out=(1,2,3),  # Output normalization dimensions
        Batchnorm=False,  # Disable batch normalization
        epochs=epochs,  # Set the number of epochs
        a=1,  # Loss function parameter, default=1
        b=1,  # Loss function parameter, default=1
        all_neurons=False,  # Whether to use all neurons for classification
        NL=NL,  # Number of layers in the model
        Layer_out=[0,1,2],  # Output layers used for classification
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
            model_path = f'./results/params_STL_l{i}.pth'
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
    loaders = get_train(batchsize=100, augment="no", Factor=1)

    # Call the main function with parsed arguments
    tsacc = main(
        epochs=args.epochs,
        device_num=args.device_num,
        tr_and_eval=args.tr_and_eval,
        save_model=args.save_model,
        loaders=loaders,
        NL=args.NL
    )