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
    #lr, epochs, lamda, lamda2, lambda_reg, lambda_covar, cutoffep, device_num

    parser = argparse.ArgumentParser(description="Pretrain a CNN using contrastiveFF", add_help=False)

    # Data 
    parser.add_argument("--augment", type=str, default='no',
                        help='Dataaugmentation or not, choose from no, single, dual')


    # Optim
    parser.add_argument("--epochs", type=int, default=20,
                        help='Number of epochs')
    parser.add_argument("--lr", type=float, default=0.018,
                        help='Base learning rate')
    parser.add_argument("--cutoffep", type=int, default=5,
                        help='After cutoffep the learning rate decays linearly to zero, Used only for linearRate decay')
    parser.add_argument("--tr_and_eval", action='store_true',
                        help='train while evaluating')
    parser.add_argument("--period", type=int, default=1000,
                        help='update the scheduler every period batches')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='exponential decay rate')
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help='weight_decay rate')
    parser.add_argument("--out_dropout", type=float, default=0.0,
                        help='out_dropout') 
    
    # Loss
    parser.add_argument("--th1", type=int, default=9,
                        help='thre1 for positive samples')
    parser.add_argument("--th2", type=int, default=1,
                        help='thre2 for negative samples')
    parser.add_argument("--lamda", type=float, default=0.02,
                        help='L2 norm regularization loss coefficient')
    parser.add_argument("--lamda2", type=float, default=0.2,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--lambda_covar", type=float, default=0.5,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--lambda_reg", type=float, default=0.0,
                        help='Weights regularization loss coefficient')
    parser.add_argument("--lambda_weight", type=float, default=0.0,
                        help='Negative samples loss coefficient')
    parser.add_argument("--p", type=int, default=1,
                        help='Number of negative samples for each postive')

    parser.add_argument("--yita", type=int, default=2,
                        help='yita')

    parser.add_argument('--device_num',type=int, default=0,
                        help='device to use for training / testing')
    parser.add_argument('--factor',type=float, default=0.9,
                        help='division factor of the training set')
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

    #mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, val_size])

    
    train_loader = DataLoader(mnist_train, batch_size= batchsize, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size= batchsize, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size= 1000, shuffle=False)
    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader, sup_trainloader


def get_pos_neg_batch_imgcats(batch_pos1, batch_pos2, p = 2):

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



def get_pos_neg_batch_imgcats_sup0(x1, x2, targets, p = 2):
    # group the data with the same targets together into lists
    Batch_lists1 = [] # 10 lists that group togher the same class
    Batch_lists2 = []

    for i in range(0, 10):
        indexes = (targets == i).nonzero(as_tuple=True)[0]
        Batch_lists1.append(x1[indexes])
        Batch_lists2.append(x2[indexes])

    # filter out the lists that contain empty values
    filtered_batch_list1 = [tensor for tensor in Batch_lists1 if tensor.nelement() > 0]
    filtered_batch_list2 = [tensor for tensor in Batch_lists2 if tensor.nelement() > 0]
    
    classes = torch.arange(0, len(filtered_batch_list1))
    # number of images in each class
    # create the postive pairs and negative pairs
    batch_poses = []
    batch_negs = []

    for i, (batch1, batch2) in enumerate(zip(filtered_batch_list1, filtered_batch_list2)):

        # random select a number inside the class
        random_indices = (torch.randperm(len(batch1)))
        #batch_pos = torch.cat((batch1, batch2[random_indices]), dim = 1)
        batch_pos = torch.cat((batch1, batch2), dim = 1)
        #print(len(batch1))
        
        shifted_idx = classes.roll(shifts=-i)[1:]
        selected_tensors = [filtered_batch_list2[idx] for idx in shifted_idx]
        # Concatenate the selected tensors
        neg_all = torch.cat(selected_tensors)
        #neg_all = torch.cat(filtered_batch_list2[classes.roll(shifts=-i)[1:].long()])
        #random_ids = torch.randint(0, len(neg_all), (len(batch1),))
        #batch_neg = torch.cat((batch1, neg_all[random_ids]), dim = 1)

        random_ids = torch.randint(0, len(neg_all), (len(batch1), p))
        batch_neg = torch.cat([torch.cat((batch1, neg_all[random_ids[:, i]]), dim=1) for i in range(p)])
        

        batch_poses.append(batch_pos)
        batch_negs.append(batch_neg)

    return torch.cat(batch_poses), torch.cat(batch_negs)


def stdnorm (x, dims = [1,2,3]):

    x = x - torch.mean(x, dim=(dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(dims), keepdim=True))

    return x

def l2norm (x):

    x = x/(x.norm(2, 1, keepdim=True) + + 1e-10)
    #x = x - torch.mean(x, dim=(dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(dims), keepdim=True))

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
    def __init__(self, in_features, out_features, Norm, droupout = 0.0, act = 0, bias=True, concat = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        if act == 0:
            self.act = torch.nn.ReLU()
        elif act == 1:
            self.act = triangle()
        
        self.relu = torch.nn.ReLU()
        #self.plas = KWinnerTakeAllpls(k=k)
        self.plas = torch.nn.ReLU()

        #self.lr = lr
        #self.opt = Adam(self.parameters(), lr=self.lr)
        self.dropout = nn.Dropout(droupout)
        self.concat = concat
        #self.threshold = threshold
        #self.num_epochs = num_epochs
        #self.btsz = batch_size
        #self.loss = []
        #self.good = []
        #self.bad = []
        if Norm == "L2norm":
            self.norm = L2norm(dims = [1])
        else:
            self.norm = standardnorm(dims = [1])

    def forward(self, x):
        x_direction =self.norm(x)

        #x_direction = x
        '''
        x = self.act(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))
        '''
        if self.concat:
            lenchannel = x.size(1)//2
            x = torch.mm(x_direction[:,:lenchannel], self.weight.T
                         ) + torch.mm(x_direction[:,lenchannel:], self.weight.T) + 2*self.bias
        else:
            x = torch.mm(x_direction, self.weight.T) + self.bias
        #x = self.dropout(x)

        return x


def orthogonal_reg(net, lambda_reg=0.01):
    # Reshape the weight tensor to shape [96, *]
    num_filters = net.conv_layer.out_channels
    w = net.conv_layer.weight.view(num_filters, -1)
    
    # Compute the Gram matrix
    gram = torch.mm(w, w.t())
    
    # Identity matrix
    identity = torch.eye(num_filters).to(w.device)
    
    # Regularization loss
    loss = lambda_reg * torch.norm(gram - identity, p='fro')
    
    return loss


def covar_reg(Z):
    # Step 1: Compute the mean vector
    z_mean = torch.mean(Z, dim=0, keepdim=True)  # Shape: (1, dims)

    # Step 2: Compute the covariance matrix
    Z_centered = Z - z_mean
    cov_matrix = (Z_centered.t() @ Z_centered) / (Z.size(0) - 1)  # Using matrix multiplication for summation. Shape: (dims, dims)

    # Step 3: Compute the covariance regularization term
    off_diagonal = cov_matrix**2 - torch.diag(cov_matrix.diag()**2)  # Squared covariance matrix with diagonal set to zero
    #regularization = off_diagonal.sum() / Z.size(1)  # Sum all entries and scale by factor 1/d
    regularization = off_diagonal.mean()

    return regularization


def contains_nan(model):
    for p in model.parameters():
        if torch.isnan(p).any():
            return True
    return False


def get_logsumexp_loss(states, temperature):
    scores = torch.matmul(states, states.t())  # (bsz, bsz)
    bias = torch.log(torch.tensor(states.shape[1], dtype=torch.float32))  # a constant
    return torch.mean(
        torch.logsumexp(scores / temperature, dim=1) - bias)

def sort(x):
    return torch.sort(x, dim=1)[0]

def get_swd_loss(states, rand_w, prior='normal', stddev=1., hidden_norm=True):
    states = torch.matmul(states, rand_w)
    states_t = sort(states.t())  # (dim, bsz)

    if prior == 'normal':
        states_prior = torch.randn_like(states) * stddev
    elif prior == 'uniform':
        states_prior = torch.rand_like(states) * 2 * stddev - stddev
    else:
        raise ValueError(f'Unknown prior {prior}')
    if hidden_norm:
        states_prior = nn.functional.normalize(states_prior, dim=-1)
    states_prior = torch.matmul(states_prior, rand_w)
    states_prior_t = sort(states_prior.t())  # (dim, bsz)

    return torch.mean((states_prior_t - states_t)**2)


def train(nets, device, optimizers,schedulers, threshold1,threshold2, epochs
            , a,b, lamda, cout,freezelayer,period,tr_and_eval, Layer_out,trainloader
            , valloader, testloader, suptrloader, out_dropout,
            search,p,pre_std,norm,aug):

    #trainloader, testloader = get_train(batchsize)

    trainouts = []; testouts = [];all_pos = [];all_neg= []
    NL = len(nets)
    #print(NL)
    #thres = [];
    meanstates=[]

    for i in range(NL):
        trainouts.append([])
        testouts.append([])
        all_pos.append([])
        all_neg.append([])
        #thres.append(torch.zeros((1,nets[i].output_channels,1,1), requires_grad=False).to(device))
        meanstates.append(0.5*torch.ones(nets[i].out_features, requires_grad = False ).to(device))
        
    #tic = time.time()
    firstpass=True
    nbbatches = 0
    #wsav = []

    traintargets = []; testtargets = [] 

    NBLEARNINGEPOCHS = epochs

    if cout:
        N_all = NBLEARNINGEPOCHS + 2
    else:
        if epochs == 0:
            N_all = NBLEARNINGEPOCHS + 1
        else:
            N_all = NBLEARNINGEPOCHS

    #NORMW  =NORMW

    #threshold = threshold

    #optimizer = Adam(net.parameters(), lr=lr)

    #criterion = nn.BCEWithLogitsLoss()
    Dims = []
    taccs = []
    # Start the experiment !

    for epoch in range(N_all):

        print("Epoch", epoch)
        #correct=0; total=0
        if epoch < NBLEARNINGEPOCHS and epochs !=0:
            for net in nets:
                net.train()
                if torch.cuda.device_count() > 1:
                    net = torch.nn.DataParallel(net)

            # Hebbian learning, unlabeled
            print("Unlabeled.")
            TESTING=False; TRAINING = False;UNLAB = True; 
            zeloader = trainloader
        else: # epoch  < UNLABPERIOD + TRAINPERIOD:
            for net in nets:
                net.eval()
                if torch.cuda.device_count() > 1:
                    net = torch.nn.DataParallel(net)
            if epoch == NBLEARNINGEPOCHS:
                # With frozen weights, acquire network responses to training set
                TESTING=False; TRAINING = True; UNLAB = False; 
                zeloader = trainloader
                print("Training top layer only!")
            elif epoch == NBLEARNINGEPOCHS + 1:
                # With frozen weights, acquire network responses to training set 
                TESTING=True; TRAINING = False; UNLAB = False
                zeloader=testloader
                print("Testing...")
            else:
                raise(ValueError("Wrong epoch!")) 


        goodness_pos = 0
        goodness_neg = 0

        for numbatch, (x,x_aug1,x_aug2, targets) in enumerate(zeloader):
            #print(numbatch)
            nbbatches += 1

            xs = []

            with torch.no_grad():

                
                x = x.to(device)
                x_aug1 = x_aug1.to(device)
                x_aug2 = x_aug2.to(device)
                #x = x - torch.mean(x, dim=(1, 2,3), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(1, 2,3), keepdim=True))

                #x = x / (x.norm(p=2, dim=[1,2,3], keepdim=True) + 1e-10)
                #x_aug1 = x_aug1 / (x_aug1.norm(p=2, dim=[1,2,3], keepdim=True) + 1e-10)
                #x_aug2= x_aug2 / (x_aug2.norm(p=2, dim=[1,2,3], keepdim=True) + 1e-10)
                
                if pre_std:
                    x = stdnorm(x, dims = [1])
                    x_aug1 = stdnorm(x_aug1, dims = [1])
                    x_aug2 = stdnorm(x_aug2, dims = [1])
                
                
                if UNLAB:
                    #x, x_neg = get_pos_neg_batch_imgcats(x_aug1, x_aug1, p=1)
                    #x, x_neg = get_pos_neg_batch_imgcats(x, x, p=1)
                    #x_, x_neg = get_pos_neg_batch_imgcats(x_aug1, x_aug2, p=9)
                    #x__, _ = get_pos_neg_batch_imgcats(x, x, p=9)
                    #x = torch.cat((x_, x__[:batchsize]))
                    #x, x_neg = get_pos_neg_batch_imgcats_sup0(x, x, targets)
                    if aug == 0:
                        x, x_neg = get_pos_neg_batch_imgcats(x, x, p= p)
                    if aug == 1:
                        x, x_neg = get_pos_neg_batch_imgcats(x_aug1, x_aug1, p= p)
                    else:
                        x, x_neg = get_pos_neg_batch_imgcats(x_aug1, x_aug2, p= p)
                    #x = x[:batchsize]
                    #x = x[:]
                    #x_neg = x_neg[:]
                    #print(x.shape, x_neg.shape)

                if TRAINING or TESTING:
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=1)
                    #, x_neg = get_pos_neg_batch_imgcats_sup(x, x, targets, p=1)
                    #targets = targets.to(device)
                    x = x[:batchsize]
                    #x_neg = x_neg[:len(x)]
                    targets = targets.to(device)
                    #print(x.shape, x_neg.shape)

            # record goodness
            # 
            # Now run the layers in succession


            for i in range(NL):

                #print("Layer " + str(i))
                optimizers[i].zero_grad()
                xs.append(x.clone())
                x = nets[i](x)
                x_neg = nets[i](x_neg)

                '''
                if triact:
                    x = x - torch.mean(x, dim = (1,), keepdim = True)
                    x_neg = x_neg- torch.mean(x_neg, dim = (1,), keepdim = True)
                '''
                xforgrad = nets[i].relu(x)
                xforgrad_neg = nets[i].relu(x_neg)

                yforgrad = xforgrad.pow(2).mean([1])
                yforgrad_neg =xforgrad_neg.pow(2).mean([1])

                #print(yforgrad.mean([0]), yforgrad_neg.mean([0]))
                
                if i < freezelayer and not (TRAINING or TESTING):
                    UNLAB = False
                    
                if i >= freezelayer and not (TRAINING or TESTING):
                    UNLAB = True

                if UNLAB :
                    
                    
                    #print(covar_reg(yfor_reg))
                    #print(covar_reg(yfor_reg))
                    """
                    if dist == 'logsumexp':
                        loss_dist_match = get_logsumexp_loss(yfor_reg_neg, temperature = 1.0)
                    else:
                        rand_w = nn.init.orthogonal_(torch.empty(hidden_dim, hidden_dim, device=device))
                        loss_dist_match = get_swd_loss(yfor_reg_neg, rand_w,
                                                    prior=dist,
                                                    hidden_norm= hidden_norm)
                    """
                    #meanstates[i] = 0.9*meanstates[i] + 0.1*(x.mean([0])).data

                    loss1 =  torch.log(1 + torch.exp(
                        a*(- yforgrad  + threshold1[i]))).mean(
                        ) + torch.log(1 + torch.exp(
                            b*(yforgrad_neg  - threshold2[i]))).mean(
                            ) +  lamda[i] * torch.norm(yforgrad[:,None], p=2, dim = (1)).mean() #+ lamda2[i]*(F.relu(yita - x.std(dim = 0)).mean()) + lambda_covar*covar_reg(x
                            #) + lamda[i] * ((meanstates[i].mean()- meanstates[i])**2).mean(
                            #)
                            #lamda[i] * torch.norm(yforgrad, p=2, dim = (1)).mean()
                            # +  orthogonal_reg(nets[i], lambda_reg=lambda_reg) 
                    

                    loss1.backward()
                    
                    optimizers[i].step()  

                    if (nbbatches+1)%period[i] == 0:
                        schedulers[i].step()
                        print(f'nbbatches {nbbatches+1} learning rate: {schedulers[i].get_last_lr()[0]}')  

                    """
                    if NORMW:
                        # Weight kept to norm 1
                        # w has shape OutChannels, InChannels, H, W
                        net.conv_layer.weight = torch.nn.Parameter(net.conv_layer.weight / (net.conv_layer.weight.norm(p=2, dim=(-3, -2, -1), keepdim=True) + 1e-8))
                        #w[numl].data =  w[numl].data / (1e-10 + torch.sqrt(torch.sum(w[numl].data ** 2, dim=[1,2,3], keepdim=True)))
                    """
                x = nets[i].act(x)
                x_neg = nets[i].act(x_neg)
                x = x.detach()
                x_neg = x_neg.detach()

                if firstpass:
                    print("Layer", i, ": x.shape:", x.shape, "y.shape (after MaxP):", x.shape, end=" ")
                    _, w = x.shape
                    Dims.append(w)
                if contains_nan(net):
                    raise optuna.TrialPruned()
             

            firstpass = False
            goodness_pos += (torch.mean(yforgrad)).item()
            #print(yforgrad.shape, yforgrad_neg.shape)
            goodness_neg += (torch.mean(yforgrad_neg)).item()

            if UNLAB and numbatch == len(zeloader) - 1:

                print(goodness_pos, goodness_neg)
                all_pos[i].append(goodness_pos)
                all_neg[i].append(goodness_neg)
                goodness_pos,  goodness_neg = 0,0

            # If we are in the phase of data collection for training/testing the linear classifier (see below)...
            
            if TRAINING or TESTING:
                # We simply collect the outputs of the network, as well as the labels. The actual training/testing occurs below with a linear classifier.
                
                if TESTING:
                    testtargets.append(targets.data.cpu().numpy())
                    for i in range(len(xs)):
                        if i < NL-1:
                            xl = xs[i+1]
                            #testouts[i].append(xl.data.cpu().numpy())
                        else:
                            xl = x
                            
                        '''
                        if not all:
                            result = extra_pool[i](xl)
                        '''
                        #result = xl
                        #if stdnorm_out:
                        result = norm(xl)
                        #result = result/result.norm(p=2, dim=dims_out, keepdim=True)
                        testouts[i].append((xl).data.cpu().numpy())

                if TRAINING:
                    traintargets.append(targets.data.cpu().numpy())
                    for i in range(len(xs)):
                        if i < NL-1:
                            xl = xs[i+1]
                            #trainouts[i].append(xl.data.cpu().numpy())
                        else:
                            xl = x
                        """
                        if not all:
                            result = extra_pool[i](xl)
                        """
                        #print(result.shape)
                        #if stdnorm_out:
                        result = norm(xl)
                        #result = result/result.norm(p=2, dim=dims_out, keepdim=True)
                        #print(result)
                        trainouts[i].append((result).data.cpu().numpy())

        
        if tr_and_eval:
            #nets_copy = deepcopy(nets)
            #torch.manual_seed(42)
            if epoch>3 and epoch%1==0:
                tacc = evaluation_(nets, Dims, device, Layer_out, out_dropout
                                   ,sup_gamma,sup_period,search, valloader, testloader, suptrloader, pre_std, norm)
    
                
                taccs.append(tacc)
            
        

    print("Training done..")
    
    if tr_and_eval:
        return nets, trainouts, testouts, traintargets, testtargets, all_pos, all_neg, Dims, taccs
    else:
        return nets, trainouts, testouts, traintargets, testtargets, all_pos, all_neg, Dims

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


def process_batch(nets, x, Layer, norm):
    outputs = []
    for j, net in enumerate(nets):
        net.eval()
        with torch.no_grad():
            x = net(x)
            x = net.act(x).detach()
            out = norm(x)
            if j in Layer:
                outputs.append(out)
    return torch.cat(outputs, dim=1)

def evaluation_(nets, dims, device, Layer_out, out_dropout, sup_gamma, sup_period, search, valloader, tsloader, suptrloader, pre_std, norm):
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


def create_lr_lambda(epochs, initial_learning_rate, final_learning_rate,cutoffep):
    # This is the lambda function that will be returned
    def lr_lambda(epoch):
        # During the first half of the epochs, return 1 (no change to the lr)
        if epoch < cutoffep:
            return 1
        # During the second half, linearly decay the lr
        else:
            decay_rate = (initial_learning_rate - final_learning_rate) / (epochs - cutoffep+1)
            return max(0, 1 - (epoch + 1 - cutoffep) * decay_rate / initial_learning_rate)
    return lr_lambda

def create_layer(layer_config,opt_config, load_params, device):
    layer_num = layer_config['num']-1

    net = Layer(layer_config["ch_in"], layer_config["channels"]
    , bias=True, Norm = "stdnorm",act = layer_config["act"])
    
    if load_params:
        net.load_state_dict(torch.load('./results/params_MNIST_l' + str(layer_num) +'.pth'))
        for param in net.parameters():
            param.requires_grad = False

    net.to(device)
    optimizer = AdamW(net.parameters(), lr=opt_config["lr"], weight_decay=opt_config["weight_decay"])
    scheduler = ExponentialLR(optimizer, opt_config["gamma"])

    return net, optimizer, scheduler

def hypersearch(epochs , a, b, bias
    ,cout, NL, Layer_out, tr_and_eval 
    , stdnorm_out 
    ,search, device_num, loaders,p, pre_std,Norm, seed_num):

    #trainloader, _, testloader,_ = get_train(batchsize, augment, Factor)
    trainloader, valloader, testloader, suptrloader = loaders

    #torch.manual_seed(1234)
    torch.manual_seed(seed_num)
    device = 'cuda:' + str(device_num) if torch.cuda.is_available() else 'cpu'
    nets = []; optimizers = []; schedulers= []; threshold1 = []; threshold2 = []; lamda = []; period= []
    trainouts = []; testouts = []; 
    #NL = len(channels)
    #print(NL)
    #lr = lr
    #NORMW = False
    #pool = []; extra_pool = []
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
        nets, _, testouts, _, _, all_pos, all_neg, _, tacc = train(
        nets, device, optimizers,schedulers, threshold1,threshold2, epochs
            , a,b, lamda,cout,freezelayer,period,tr_and_eval, Layer_out,trainloader
            , valloader, testloader, suptrloader, opt_config['out_dropout'],
            search,p, pre_std, norm, layer_config["aug"])

    else:
        nets, trainouts, testouts, traintargets, testtargets, all_pos, all_neg, Dims = train(
            nets, device, optimizers,schedulers, threshold1,threshold2, epochs
            , a,b, lamda, cout,freezelayer,period,tr_and_eval, Layer_out,trainloader
            , valloader, testloader, suptrloader, opt_config['out_dropout'],
            search,p, pre_std, norm,layer_config["aug"])
        #torch.manual_seed(42)

        tacc = evaluation_(nets, Dims, device, Layer_out
            ,opt_config['out_dropout'],sup_gamma,sup_period,search,valloader, testloader, suptrloader, pre_std, norm)
        #tacc = evaluation(trainouts, testouts, traintargets, testtargets, Layer_out)
        #tacc = evaluation_qudrant(trainouts, testouts, traintargets, testtargets, device, Layer_out)

    return tacc, all_pos, all_neg, testouts, nets


def main(epochs, device_num,tr_and_eval 
         ,save_model, loaders, NL):

    tacc, all_pos, all_neg, testouts, nets = hypersearch(
    epochs = epochs, 
    a = 1.0,
    b = 1.0,
    bias = True,
    cout = False,
    NL = NL,
    Layer_out = [0,1],
    tr_and_eval = tr_and_eval,
    stdnorm_out = "stdnorm",
    Norm = "stdnorm", #L2norm, stdnorm, no
    search = False,
    device_num = device_num,
    loaders = loaders,
    p = 1,
    pre_std = True,
    seed_num = 1234)

    # save the model
    if save_model:
        for i, net in enumerate(nets):
            torch.save(net.state_dict(), './results/params_l'+str(i)+'_sup.pth')

    return tacc




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('ContrastFF script', parents=[get_arguments()])
    args = parser.parse_args()
    
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    
    #{'lr': 0.018, 'lamda': 0, 'lamda2': 0.644632082149124, 'th1': 3, 'th2': 5, 
    #'lambda_covar': 3.5385363579299187, 'lambda_weight': 5.315934820285312, 'gamma': 0.9, 
    #'yita': 1, 'out_dropout': 0, 'period': 2000, 'weight_decay': 0.0003, 'epochs': 19, 'p': 5}
    
    loaders = get_train(batchsize=100, augment=True)
    tsacc =  main(epochs=args.epochs, device_num=args.device_num, tr_and_eval=args.tr_and_eval
     , save_model = args.save_model, loaders=loaders, NL = args.NL)
    