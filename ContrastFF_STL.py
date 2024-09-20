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
    parser.add_argument("--yita", type=int, default=2,
                        help='yita')
    parser.add_argument("--p", type=int, default=1,
                        help='Number of negative samples for each postive')
    parser.add_argument("--tau", type=float, default=1.,
                        help='temperature')

    parser.add_argument('--device_num',type=int, default=0,
                        help='device to use for training / testing')
    parser.add_argument('--factor',type=float, default=0.9,
                        help='division factor of the training set')
    parser.add_argument("--save_model", action='store_true',
                        help='save model or not')
    parser.add_argument("--NL", type=int, default=1,
                        help='Number of layers')                    


    return parser


#custom the trainloader to include the augmented views of the original batch
torch.manual_seed(1234)
# Define the two sets of transformations
BATCHSIZE = 50
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
    def __init__(self, root, augment="No", *args, **kwargs):
        super(DualAugmentSTL10, self).__init__(root,*args, **kwargs)
        self.augment = augment
        
    def __getitem__(self, index):

        #if self.labels is not None:
            #img, target = self.data[index], int(self.labels[index])
        #else:
            #img, target = self.data[index], None

        img, target = self.data[index], self.labels[index]
        
        #img = torch.tensor(img, dtype=torch.float)
        img_pil = ToPILImage()(img.transpose(1, 2, 0))
        img_original = transform_train(img_pil)

        #print(img_original.shape)

        if self.augment == "single":
            img1 = transform1(img_pil)
            return img_original, img1, img_original, target
        elif self.augment == "dual":
            img1 = transform1(img_pil)
            img2 = transform2(img_pil)
            return img_original, img1, img2, target
        else:
            return img_original, img_original, img_original, target

            
class STL10_test(STL10):
    def __init__(self, aug=False, *args, **kwargs):
        super(STL10_test, self).__init__(*args, **kwargs)
        self.aug = aug
        
    def __getitem__(self, index):
        #img, target = self.data[index], self.targets[index]
        #if self.labels is not None:
            #img, target = self.data[index], int(self.labels[index])
       #else:
            #img, target = self.data[index], None
        img, target = self.data[index], self.labels[index]
        # Convert the image to PIL format
        #print(img.shape)
        #img = torch.tensor(img, dtype=torch.float)
        img = ToPILImage()(img.transpose(1, 2, 0))
        
        # Apply the two sets of transformations
        if self.aug:
            img = transform_train(img)
        else:
            img = transform_test(img)
        #img2 = transform(img)
        
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


def get_pos_neg_batch_imgcats_sup(x1, x2, targets, p = 2):
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
        #random_indices = (torch.randperm(len(batch1)))
        #batch_pos = torch.cat((batch1, batch2[random_indices]), dim = 1)
        #print(len(batch1))
        if len(batch1) < 2:
            batch_pos = torch.cat((batch1, batch2), dim = 1)
        else:
            random_indices =  torch.randint(1, len(batch1), (1,))
            labeles = torch.arange(len(batch1))
            batch_pos = torch.cat((torch.cat((batch1, batch2), dim = 1),
                                torch.cat((batch1, batch2[(labeles+random_indices)%len(batch1)]), dim = 1)))

        shifted_idx = classes.roll(shifts=-i)[1:]
        selected_tensors = [filtered_batch_list2[idx] for idx in shifted_idx]
        # Concatenate the selected tensors
        neg_all = torch.cat(selected_tensors)
        #neg_all = torch.cat(filtered_batch_list2[classes.roll(shifts=-i)[1:].long()])
        #random_ids = torch.randint(0, len(neg_all), (len(batch1),))
        #batch_neg = torch.cat((batch1, neg_all[random_ids]), dim = 1)

        random_ids = torch.randint(0, len(neg_all), (len(batch1), p))
        batch_neg = torch.cat((torch.cat((batch1, neg_all[random_ids[:, 0]]), dim = 1),
                               torch.cat((batch1, neg_all[random_ids[:, 1]]), dim = 1)))
        

        batch_poses.append(batch_pos)
        batch_negs.append(batch_neg)

    return torch.cat(batch_poses), torch.cat(batch_negs)

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
        batch_pos = torch.cat((batch1, batch2[random_indices]), dim = 1)
        #print(len(batch1))
        

        shifted_idx = classes.roll(shifts=-i)[1:]
        selected_tensors = [filtered_batch_list2[idx] for idx in shifted_idx]
        # Concatenate the selected tensors
        neg_all = torch.cat(selected_tensors)
        #neg_all = torch.cat(filtered_batch_list2[classes.roll(shifts=-i)[1:].long()])
        #random_ids = torch.randint(0, len(neg_all), (len(batch1),))
        #batch_neg = torch.cat((batch1, neg_all[random_ids]), dim = 1)

        random_ids = torch.randint(0, len(neg_all), (len(batch1), p))
        batch_neg = torch.cat((torch.cat((batch1, neg_all[random_ids[:, 0]]), dim = 1),
                               torch.cat((batch1, neg_all[random_ids[:, 1]]), dim = 1)))
        

        batch_poses.append(batch_pos)
        batch_negs.append(batch_neg)

    return torch.cat(batch_poses), torch.cat(batch_negs)


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
    def __init__(
            self, input_channels, output_channels, kernel_size, pad = 0, batchnorm = False, normdims = [1,2,3],norm = "stdnorm",
            bias = True, dropout = 0., padding_mode = 'reflect', concat = True, act = 'relu'):
        super(Conv2d, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.normdims = normdims

        # Weights: [output_channels, 32, 32, input_channels, kernel_size[0], kernel_size[1]]
        self.conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, bias=bias)
        init.xavier_uniform_(self.conv_layer.weight)
        #weight_range = 25 / math.sqrt(input_channels * kernel_size[0] * kernel_size[1])
        #self.conv_layer.weight.data = nn.Parameter(weight_range * torch.randn((output_channels, input_channels, kernel_size[0], kernel_size[1])))
        self.padding_mode = padding_mode# zeros
        self.F_padding = (pad, pad, pad, pad)
        
        if act == 'relu':
            self.act = torch.nn.ReLU()
        else:
            self.act = triangle()
            #x.data =  x.data - torch.mean(x.data, axis=1, keepdims=True)
        #self.posact = ThresholdedReLU(threshold = reth, power = actp)
        #self.act = CustomActivation(alpha=alpha, theta = theta)
        self.dropout = nn.Dropout(p=dropout)  # 50% dropout
        self.relu = torch.nn.ReLU()
        self.concat = concat
        #self.act = temp_scaled_sigmoid(T = t)
        #self.act = AbsActivation()
        #self.preact = temp_scaled_sigmoid(T = t)
        #self.power = actp
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(self.input_channels, affine=False)
        else:
            self.bn1 = nn.Identity()

        if norm == "L2norm":
            self.norm = L2norm(dims = normdims)
        elif norm == "stdnorm":
            self.norm = standardnorm(dims = normdims)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        # Dynamic asymmetric padding
        
        x = self.bn1(x)
        
        x = F.pad(x, self.F_padding, self.padding_mode)
        #x = x / (x.norm(p=2, dim=(self.normdims), keepdim=True) + 1e-10)
        x = self.norm(x)
        
        if self.concat: 
            lenchannel = x.size(1)//2
            out = self.conv_layer(x[:, :lenchannel]) + self.conv_layer(x[:, lenchannel:])
        else:
            out = self.conv_layer(x)
        #out = out - torch.mean(out, dim=(1, 2,3),keepdim=True)#;  out = out / (1e-10 + torch.std(out, dim=(1, 2,3), keepdim=True)) 

        return out
    
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


def train(nets, device, optimizers,schedulers, threshold1,threshold2,  dims_in, dims_out, epochs, pool
            , a,b, lamda, freezelayer,period,extra_pool, tr_and_eval, Layer_out, all,trainloader
            , valloader, testloader, suptrloader,pre_std, stdnorm_out, search,  Factor, p
            , config):


    #trainloader, testloader = get_train(batchsize)

    trainouts = []; testouts = [];all_pos = [];all_neg= []
    NL = len(nets)
    #print(NL)
    thres = [];meanstates=[]

    for i in range(NL):
        trainouts.append([])
        testouts.append([])
        all_pos.append([])
        all_neg.append([])
        thres.append(torch.zeros((1,nets[i].output_channels,1,1), requires_grad=False).to(device))
        meanstates.append(0.5*torch.ones(nets[i].output_channels, requires_grad = True ).to(device))
        
    #tic = time.time()
    firstpass=True
    nbbatches = 0
    #wsav = []

    traintargets = []; testtargets = [] 

    NBLEARNINGEPOCHS = epochs

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
            nets[-1].train()
            #if torch.cuda.device_count() > 1:
                #net = torch.nn.DataParallel(net)

            # Hebbian learning, unlabeled
            print("Unlabeled.")
            TESTING=False; TRAINING = False;UNLAB = True; 
            zeloader = trainloader
        else: # epoch  < UNLABPERIOD + TRAINPERIOD:
            for net in nets:
                net.eval()
                #if torch.cuda.device_count() > 1:
                    #net = torch.nn.DataParallel(net)
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

        for numbatch, (x,_,_, targets) in enumerate(zeloader):
            #print(numbatch)
            nbbatches += 1

            xs = []

            #with torch.no_grad():

                
            x = x.to(device)
            
            for i in range(NL):

                #print("Layer " + str(i))
                #optimizers[i].zero_grad()
                if nets[i].concat:
                    x = stdnorm(x, dims = dims_in)
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)
                    
                xs.append(x.clone())
                x = nets[i](x)
                x_neg = nets[i](x_neg)

                xforgrad = nets[i].relu(x)
                xforgrad_neg = nets[i].relu(x_neg)
                prepool = xforgrad

                yforgrad = xforgrad.pow(2).mean([1])
                yforgrad_neg =xforgrad_neg.pow(2).mean([1])

                #print(yforgrad.mean([0]), yforgrad_neg.mean([0]))

                #x = net.act(x).detach()
                #x_neg = net.act(x_neg).detach()

                xforgrad = pool[i](xforgrad)
                xforgrad_neg = pool[i](xforgrad_neg)

                #calculate the final output for regularization
                yfor_reg = extra_pool[i](xforgrad).view(xforgrad.shape[0], -1)

                

                if i < freezelayer and not (TRAINING or TESTING):
                    UNLAB = False
                    
                if i >= freezelayer and not (TRAINING or TESTING):
                    UNLAB = True

                if UNLAB :
                    
                    #meanstates[i] = 0.9*meanstates[i].data + 0.1*(prepool.mean([-1,-2]).mean([0]))
                    
                    optimizers[i].zero_grad()
                    loss1 =  torch.log(1 + torch.exp(
                        a*(- yforgrad  + threshold1[i]))).mean([1,2]).mean(
                        ) + torch.log(1 + torch.exp(
                            b*(yforgrad_neg  - threshold2[i]))).mean([1,2]).mean() + lamda[i] * torch.norm(yforgrad, p=2, dim = (1,2)).mean(
                            ) #+ lamda2[i]*F.relu(yita-yfor_reg.std(dim = 0)).mean(
                            #) #+ lambda_covar*covar_reg(yfor_reg
                            #)
                    

                    loss1.backward()
                    
                    optimizers[i].step()  

                    if (nbbatches+1)%period[i] == 0:
                        schedulers[i].step()
                        print(f'nbbatches {nbbatches+1} learning rate: {schedulers[i].get_last_lr()[0]}')  


                x = pool[i](nets[i].act(x))
                x_neg = pool[i](nets[i].act(x_neg))

                x = x.detach()
                x_neg = x_neg.detach()

                if firstpass:
                    print("Layer", i, ": x.shape:", x.shape, "y.shape (after MaxP):", x.shape, end=" ")
                    _, channel, h, w = x.shape
                    Dims.append(channel * h * w)
                if contains_nan(nets[i]):
                    raise optuna.TrialPruned()
             

            firstpass = False
            goodness_pos += (torch.mean(yforgrad.mean([1,2]))).item()
            #print(yforgrad.shape, yforgrad_neg.shape)
            goodness_neg += (torch.mean(yforgrad_neg.mean([1,2]))).item()

            if UNLAB and numbatch == len(zeloader) - 1:

                print(goodness_pos/len(zeloader), goodness_neg/len(zeloader))
                #print(goodness_pos, goodness_neg)
                all_pos[i].append(goodness_pos)
                all_neg[i].append(goodness_neg)
                goodness_pos,  goodness_neg = 0,0

            # If we are in the phase of data collection for training/testing the linear classifier (see below)...
            
            #schedulers[i].step()
            #print(f'Epoch {epoch+1}/{epochs} learning rate: {schedulers[i].get_last_lr()[0]}')              

        if tr_and_eval:
            #nets_copy = deepcopy(nets)
            #torch.manual_seed(42)
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
    #model.dropout.train()
    #print(epoch)
    #for numbatch, (x,x_aug1,x_aug2, targets) in enumerate(zeloader)
    for i, (x, labels) in enumerate(loader):
        
        #x = x[:len(x)//2]

        x = x.to(config.device)
        labels = labels.to(config.device)

        outputs = []
        for j, net in enumerate(nets):
            net.eval()
            with torch.no_grad():
                if net.concat:
                    x = stdnorm(x, dims = config.dims_in)
                    x = torch.cat((x, x), dim=1)

                x = net(x)

                #x = net.act(x)
                x = pool[j](net.act(x))

                if not config.all_neurons:
                    out = extra_pool[j](x)

                if config.stdnorm_out:
                    out = stdnorm(out, dims = config.dims_out)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

                #x.data =  x.data - torch.mean(x.data, axis=1, keepdims=True)
                #x = net.act(x)
                #x = pool[j](x).detach()

        outputs = torch.cat(outputs, dim = 1)    
        #print(outputs.shape)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        #print(inputs.device, next(classifier.parameters()).device)
        outputs = classifier(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    pass

def test_readout(classifier, nets, pool, extra_pool, loader, criterion, config, epoch, mode):
    # Testing/evaluation loop implementation
    # on the test set
    classifier.eval()
    running_loss = 0.
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    #if search:
    with torch.no_grad():
        for i, (x, labels) in enumerate(loader):
        #for data in testloader:
            #images, labels = data
            
            #x = x[:len(x)//2]

            x = x.to(config.device)
            labels = labels.to(config.device)

            outputs = []
            for j, net in enumerate(nets):
                net.eval()
                #with torch.no_grad():
                if net.concat:
                    x = stdnorm(x, dims = config.dims_in)
                    x = torch.cat((x, x), dim=1)
                x = net(x)

                #x = net.act(x)
                x = pool[j](net.act(x))

                if not config.all_neurons:
                    out = extra_pool[j](x)

                if config.stdnorm_out:
                    out = stdnorm(out, dims = config.dims_out)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

                #x.data =  x.data - torch.mean(x.data, axis=1, keepdims=True)
                #x = net.act(x)
                #x = pool[j](x).detach()

            outputs = torch.cat(outputs, dim = 1) 
            # calculate outputs by running images through the network
            outputs = classifier(outputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    print(f'Accuracy of the network on the 10000 '+ mode+ f' images: {100 * correct / total} %')
    print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

    return correct / total

def evaluate_model(nets, pool, extra_pool, config, loaders, search, Dims):
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    #Layer = config.Layer_out
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
    #epochs = 100
    for epoch in range(epochs):
        train_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, optimizer, config, epoch)
        lr_scheduler.step()
        if epoch % 20 == 0 or epoch == (epochs-1):
            acc_train = test_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, config,epoch, 'Train')
            acc_val = test_readout(classifier, nets, pool, extra_pool, valloader, criterion, config,epoch, 'Val')
    torch.set_rng_state(current_rng_state)
    return acc_train, acc_val  


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

    net = Conv2d(layer_config["ch_in"], layer_config["channels"], (layer_config["kernel_size"], layer_config["kernel_size"]),
                  pad = layer_config["pad"], norm = "stdnorm", padding_mode = layer_config["padding_mode"], act = layer_config["act"])
    
    if load_params:
        net.load_state_dict(torch.load('./results/params_STL_l' + str(layer_num) +'.pth'))
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
    , a, b, all_neurons, NL,  tr_and_eval
    ,pre_std, stdnorm_out, search, device_num, Factor, loaders,p,seed_num):

    trainloader, valloader, testloader, suptrloader = loaders

    torch.manual_seed(seed_num)
    
    device = 'cuda:' + str(device_num) if torch.cuda.is_available() else 'cpu'
    nets = []; optimizers = []; schedulers= []; threshold1 = []; threshold2 = []; lamda = []; period= []
    trainouts = []; testouts = []; 
    pools = []; extra_pools = []

    with open('config.json', 'r') as f:
        config = json.load(f)

    freezelayer = NL-1
    
    for i, (layer_config, opt_config) in enumerate(zip(config['STL']['layer_configs'][:NL], config['STL']['opt_configs'][:NL])):
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
                              out_dropout=opt_config['out_dropout'], Layer_out=opt_config['Layer_out'],pre_std = pre_std, all_neurons = all_neurons)
    tacc = 0


    if tr_and_eval:
        nets, all_pos, all_neg, _, tacc = train(
        nets, device, optimizers,schedulers, threshold1,threshold2, dims_in, dims_out, epochs, pools, a,b, lamda, freezelayer
        ,period, extra_pools, tr_and_eval, opt_config['Layer_out'], all_neurons,trainloader, valloader, testloader, suptrloader,pre_std, stdnorm_out
        ,search, Factor,p, config)

    else:
        nets, all_pos, all_neg, Dims = train(
        nets, device, optimizers,schedulers, threshold1,threshold2, dims_in, dims_out, epochs, pools, a,b, lamda, freezelayer
        ,period, extra_pools, tr_and_eval, opt_config['Layer_out'], all_neurons,trainloader, valloader, testloader, suptrloader,pre_std, stdnorm_out
        ,search, Factor,p, config)
        #torch.manual_seed(42)
        tacc = evaluate_model(nets, pools, extra_pools, config, loaders, search, Dims)
        #tacc = evaluation(trainouts, testouts, traintargets, testtargets, Layer_out)
        #tacc = evaluation_qudrant(trainouts, testouts, traintargets, testtargets, device, Layer_out)
    
    return tacc, all_pos, all_neg, testouts, nets


def main(epochs, device_num,tr_and_eval 
         ,save_model, loaders, NL):
    tacc, all_pos, all_neg, testouts, nets = hypersearch(
        dims =  (1,2,3),
        dims_in = (1,2,3), 
        dims_out = (1,2,3),
        Batchnorm = False, 
        epochs = epochs, 
        a = 1,
        b = 1,
        all_neurons = False,
        NL = NL,
        tr_and_eval = tr_and_eval,
        pre_std = True,
        stdnorm_out = True,
        search = False,
        device_num = device_num,
        Factor = 1,
        loaders = loaders,
        p = 1, 
        seed_num = 1234)

    # save the model
    if save_model:
        for i, net in enumerate(nets):
            torch.save(net.state_dict(), './results/params_STL_l'+str(i)+'.pth')

    return tacc



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('ContrastFF script', parents=[get_arguments()])
    args = parser.parse_args()
    
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    
    loaders = get_train(batchsize=100, augment="no", Factor=1)
    tsacc =  main(epochs=args.epochs, device_num=args.device_num, tr_and_eval=args.tr_and_eval
     , save_model = args.save_model, loaders=loaders, NL = args.NL)