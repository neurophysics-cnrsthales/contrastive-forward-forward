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
#torch.use_deterministic_algorithms(True)



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
    parser.add_argument("--tau", type=float, default=1.,
                        help='temperature')

    parser.add_argument("--yita", type=int, default=2,
                        help='yita')

    parser.add_argument('--device_num',type=int, default=0,
                        help='device to use for training / testing')
    parser.add_argument('--factor',type=float, default=0.9,
                        help='division factor of the training set')
    parser.add_argument("--save_model", action='store_true',
                        help='save model or not')


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
            return img_original, img_original, img_original, target

            
class DualAugmentCIFAR10_test(torchvision.datasets.CIFAR10):
    def __init__(self, aug=False, *args, **kwargs):
        super(DualAugmentCIFAR10_test, self).__init__(*args, **kwargs)
        self.aug = aug
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        #print(img.shape)
        # Convert the image to PIL format
        img = ToPILImage()(img)
        #print(img.shape)
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

    #testset = DualAugmentCIFAR10_test(root='./data', train=False, download=True)
    #testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    #sup_trainset = DualAugmentCIFAR10_test(root='./data', train=True, download=True)
    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True, )

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
        #batch_neg = torch.cat((torch.cat((batch1, neg_all[random_ids[:, 0]]), dim = 1),
                               #torch.cat((batch1, neg_all[random_ids[:, 1]]), dim = 1)))
        
        batch_neg = torch.cat([torch.cat((batch1, neg_all[random_ids[:, i]]), dim=1) for i in range(p)])


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


# with padding version
class Conv2d(nn.Module):
    def __init__(
            self, input_channels, output_channels, kernel_size, pad = 0, actp = 1, batchnorm = False, normdims = [1,2,3],norm = "L2norm",
            t = 1, bias = True, reth = 0,alpha=0.1, theta = 0.5, dropout = 0.5, padding_mode = 'reflect'):
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
        
        self.act = torch.nn.ReLU()
        #self.posact = ThresholdedReLU(threshold = reth, power = actp)
        #self.act = CustomActivation(alpha=alpha, theta = theta)
        self.dropout = nn.Dropout(p=dropout)  # 50% dropout
        
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
        
        x = self.dropout(x)
        lenchannel = x.size(1)//2
        out = self.conv_layer(x[:, :lenchannel]) + self.conv_layer(x[:, lenchannel:])
        '''
        if x.shape[1] == 6: # x is input image
            out = self.conv_layer(x[:, :3]) + self.conv_layer(x[:, 3:])
        else:
            out = self.conv_layer(x)
        
        out = self.conv_layer(x)
        '''
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


def train(nets, device, optimizers,schedulers, threshold1,threshold2,  dims_in, dims_out, epochs, batchsize, pool, NORMW
            , a,b, lamda,lamda2,posact, cout,freezelayer,period,extra_pool, tr_and_eval, Layer_out, all,trainloader
            , valloader, testloader, suptrloader,pre_std, triact,stdnorm_out,triact_pos, out_dropout,yita,lambda_reg
            ,lambda_covar, sup_gamma,sup_period,search, augment, Factor, p, lambda_weight, config):


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
        #meanstates.append(0.5*torch.ones(nets[i].output_channels, 32, 32, requires_grad = False ).to(device))
        meanstates.append(0.5*torch.ones(nets[i].output_channels, requires_grad = True ).to(device))
        
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
            #for net in nets:
            nets[-1].train()
            #if torch.cuda.device_count() > 1:
                #nets[-1] = torch.nn.DataParallel(nets[-1])

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

        for numbatch, (x,x_aug1,x_aug2, targets) in enumerate(zeloader):
            #print(numbatch)
            nbbatches += 1

            xs = []

            #with torch.no_grad():

                
            x = x.to(device)
            #x_aug1 = x_aug1.to(device)
            #x_aug2 = x_aug2.to(device)
            #x = x - torch.mean(x, dim=(1, 2,3), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(1, 2,3), keepdim=True))

            #x = x / (x.norm(p=2, dim=[1,2,3], keepdim=True) + 1e-10)
            #x_aug1 = x_aug1 / (x_aug1.norm(p=2, dim=[1,2,3], keepdim=True) + 1e-10)
            #x_aug2= x_aug2 / (x_aug2.norm(p=2, dim=[1,2,3], keepdim=True) + 1e-10)

            
                #print(x.shape, x_neg.shape)

            # record goodness
            # 
            # Now run the layers in succession


            for i in range(NL):

                

                if pre_std:
                    x = stdnorm(x, dims = dims_in)
                    #x_aug1 = stdnorm(x_aug1, dims = dims_in)
                    #x_aug2 = stdnorm(x_aug2, dims = dims_in)

                    
                if TRAINING or TESTING:
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=1)
                    x = x[:batchsize]
                    #x_neg = x_neg[:len(x)]
                    targets = targets.to(device)
                    xs.append(x.clone())
                else:
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)
                #print("Layer " + str(i))
                #optimizers[i].zero_grad()
                
                x = nets[i](x)
                x_neg = nets[i](x_neg)


                xforgrad = nets[i].act(x)
                xforgrad_neg = nets[i].act(x_neg)
                
                """
                # check if there are any dead neurons
                dead_x = (x <= 0).all(dim=0).all(dim=1).all(dim=1)
                dead_x_neg = (x_neg <= 0).all(dim=0).all(dim=1).all(dim=1)
                # Get the indices of dead neurons
                dead_neuron_indices = dead_x.nonzero(as_tuple=False).squeeze()
                dead_neuron_indices_neg = dead_x_neg.nonzero(as_tuple=False).squeeze()
                print(f"Dead neuron indices: {dead_neuron_indices}")
                print(f"Dead neuron indices_neg: {dead_neuron_indices_neg}")
                """
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
                #yfor_reg_neg = extra_pool[i](x_neg).view(x_neg.shape[0], -1)

                """Generalized contrastive loss.

                Both hidden1 and hidden2 should have shape of (n, d).

                Configurations to get following losses:
                * decoupled NT-Xent loss: set dist='logsumexp', hidden_norm=True
                * SWD with normal distribution: set dist='normal', hidden_norm=False
                * SWD with uniform hypersphere: set dist='normal', hidden_norm=True
                * SWD with uniform hypercube: set dist='uniform', hidden_norm=False
                """
                """
                dist = 'normal' #'normal' 'logsumexp'
                hidden_dim = yfor_reg_neg.shape[-1]  # get hidden dimension
                hidden_norm = True
                if hidden_norm:
                    yfor_reg_neg = F.normalize(yfor_reg_neg, p=2, dim=-1)
                """

                if i < freezelayer and not (TRAINING or TESTING):
                    UNLAB = False
                    
                if i >= freezelayer and not (TRAINING or TESTING):
                    UNLAB = True

                if UNLAB :
                    optimizers[i].zero_grad()
                    #meanstates[i] = 0.9*meanstates[i].data + 0.1*(prepool.mean([-1,-2]).mean([0]))

                    #0.9*meanstates[i] + 0.1*(prepool.mean([-1,-2]).mean([0])).data
                    """
                    #print(covar_reg(yfor_reg))
                    #print(covar_reg(yfor_reg))
                    
                    if dist == 'logsumexp':
                        loss_dist_match = get_logsumexp_loss(yfor_reg_neg, temperature = 1.0)
                    else:
                        rand_w = nn.init.orthogonal_(torch.empty(hidden_dim, hidden_dim, device=device))
                        loss_dist_match = get_swd_loss(yfor_reg_neg, rand_w,
                                                    prior=dist,
                                                    hidden_norm= hidden_norm)
                    """
                    
                    loss =  torch.log(1 + torch.exp(
                        a*(- yforgrad  + threshold1[i]))).mean([1,2]).mean(
                        ) + torch.log(1 + torch.exp(
                            b*(yforgrad_neg  - threshold2[i]))).mean([1,2]).mean() + lamda[i] * torch.norm(yforgrad, p=2, dim = (1,2)).mean(
                            ) #+ lamda2[i]*(F.relu(yita - yfor_reg.std(dim = 0)).mean()) #+ lambda_covar*covar_reg(yfor_reg
                            #) #+ lambda_weight * loss_dist_match
                            # +  orthogonal_reg(nets[i], lambda_reg=lambda_reg) 
                    """

                    #print( ((meanstates[i].mean()- meanstates[i])**2).mean().item(),lamda[i])
                    #print(meanstates[i])
                    loss1 =  torch.log(1 + torch.exp(
                        a*(- yforgrad  + threshold1[i]))).mean([1,2]).mean(
                        ) + torch.log(1 + torch.exp(
                            b*(yforgrad_neg  - threshold2[i]))).mean([1,2]).mean() 
                    #meanmeanstates = meanstates[i].mean()
                    loss2 =  lamda[i] * ((meanstates[i].mean()- meanstates[i])**2).mean(
                            )
                    loss = loss1 + loss2
                    """
                    """
                    mean_diff = prepool.mean([0])
                    mean_diff.data = (meanstates[i].mean([0])- meanstates[i]).data

                    loss2 = lamda[i] * ( -1/2 *mean_diff**2).mean(
                            ) + lamda2[i]*(F.relu(yita - yfor_reg.std(dim = 0)).mean())  #+ lambda_covar*covar_reg(yfor_reg
                            #) #+ lambda_weight * loss_dist_match
                    
                    loss1 =  torch.cat((torch.log(1 + torch.exp(
                        a*(- yforgrad  + threshold1[i]))).mean([1,2]), torch.log(1 + torch.exp(
                            b*(yforgrad_neg  - threshold2[i]))).mean([1,2])), dim = 0).mean() + lamda[i] * ((meanstates[i].mean()- meanstates[i])**2).mean(
                            ) + lamda2[i]*(F.relu(yita - yfor_reg.std(dim = 0)).mean())  #+ lambda_covar*covar_reg(yfor_reg
                            #) #+ lambda_weight * loss_dist_match
                    """
                    
                    loss.backward()

                    optimizers[i].step()  

                    if (nbbatches+1)%period[i] == 0:
                        schedulers[i].step()
                        print(f'nbbatches {nbbatches+1} learning rate: {schedulers[i].get_last_lr()[0]}')  

                    if NORMW:
                        # Weight kept to norm 1
                        # w has shape OutChannels, InChannels, H, W
                        net.conv_layer.weight = torch.nn.Parameter(net.conv_layer.weight / (net.conv_layer.weight.norm(p=2, dim=(-3, -2, -1), keepdim=True) + 1e-8))
                        #w[numl].data =  w[numl].data / (1e-10 + torch.sqrt(torch.sum(w[numl].data ** 2, dim=[1,2,3], keepdim=True)))

                x.data =  x.data - torch.mean(x.data, axis=1, keepdims=True)
                x_neg.data =  x_neg.data - torch.mean(x_neg.data, axis=1, keepdims=True)

                x = nets[i].act(x)
                x_neg = nets[i].act(x_neg)

                x = pool[i](x)
                x_neg = pool[i](x_neg)

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
                            
                        if not all:
                            result = extra_pool[i](xl)
                        
                        #result = xl
                        if stdnorm_out:
                            result = stdnorm(result, dims = dims_out)
                        #result = result/result.norm(p=2, dim=dims_out, keepdim=True)
                        testouts[i].append((result).data.cpu().numpy())

                if TRAINING:
                    traintargets.append(targets.data.cpu().numpy())
                    for i in range(len(xs)):
                        if i < NL-1:
                            xl = xs[i+1]
                            #trainouts[i].append(xl.data.cpu().numpy())
                        else:
                            xl = x
                        if not all:
                            result = extra_pool[i](xl)
                        #print(result.shape)
                        if stdnorm_out:
                            result = stdnorm(result, dims = dims_out)
                        #result = result/result.norm(p=2, dim=dims_out, keepdim=True)
                        #print(result)
                        trainouts[i].append((result).data.cpu().numpy())

        
            #schedulers[i].step()
            #print(f'Epoch {epoch+1}/{epochs} learning rate: {schedulers[i].get_last_lr()[0]}')              

        if tr_and_eval:
            #nets_copy = deepcopy(nets)
            #torch.manual_seed(42)
            if epoch>3 and epoch%1==0:
                tacc = evaluate_model(nets, pool, extra_pool, config, loaders, search, Dims)
                taccs.append(tacc)
            #for net in nets:
                #net.train()
        #if (epoch+1)%period[i] == 0:
            #schedulers[i].step()
        

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
                if config.pre_std:
                    x = stdnorm(x, dims = config.dims_in)
                x = torch.cat((x, x), dim=1)

                x = net(x)
                
                out = pool[j](net.act(x)).detach()

                x.data =  x.data - torch.mean(x.data, axis=1, keepdims=True)
                x = net.act(x)
                x = pool[j](x).detach()

                if j == (len(nets)-1):
                    out = extra_pool[j](out)
                else:
                    out = extra_pool[j](x)

                if config.stdnorm_out:
                    out = stdnorm(out, dims = config.dims_out)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

                

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
                with torch.no_grad():
                    if config.pre_std:
                        x = stdnorm(x, dims = config.dims_in)
                    x = torch.cat((x, x), dim=1)
                    x = net(x)

                    out = pool[j](net.act(x)).detach()

                    x.data =  x.data - torch.mean(x.data, axis=1, keepdims=True)
                    x = net.act(x)
                    x = pool[j](x).detach()

                    if j == (len(nets)-1):
                        out = extra_pool[j](out)
                    else:
                        out = extra_pool[j](x)

                    if config.stdnorm_out:
                        out = stdnorm(out, dims = config.dims_out)
                    out = out.flatten(start_dim=1)
                    if j in config.Layer_out:
                        outputs.append(out)

                    

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
    lr_scheduler = CustomStepLR(optimizer, nb_epochs=50)
    criterion = nn.CrossEntropyLoss()

    if not search:
        valloader = testloader
    # Main evaluation loop
    for epoch in range(50):
        train_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, optimizer, config, epoch)
        lr_scheduler.step()
        if epoch % 20 == 0 or epoch == 49:
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

def hypersearch(channels, threshold1,threshold2, lr, dims, dims_in, dims_out, power, Batchnorm, epochs
    ,pool_size, batchsize, pooltype, kernel_size, t, NORMW, a, b, bias, reth,gamma,lamda,lamda2,alpha
    , theta,pad,posact,cout, all_neurons, freezelayer, Layer_out,period,extra_pool_size,stride_size,tr_and_eval,dropout,weight_decay
    ,pre_std, triact,stdnorm_out, norm, out_dropout,triact_pos, padding_mode,yita,lambda_reg, sup_gamma,sup_period,lambda_covar
    ,search, cutoffep, device_num, augment, Factor, loaders,p, lambda_weight,padding,seed_num):

    #trainloader, _, testloader,_ = get_train(batchsize, augment, Factor)
    trainloader, valloader, testloader, suptrloader = loaders

    #torch.manual_seed(1234)
    torch.manual_seed(seed_num)
    #current_rng_state = torch.get_rng_state()
    #print(current_rng_state[:10], len(current_rng_state))

    #torch.manual_seed(seed_num)
    device = 'cuda:' + str(device_num) if torch.cuda.is_available() else 'cpu'
    nets = []; optimizers = []; schedulers= []; all_pos = []; all_neg = []#;xs = []
    trainouts = []; testouts = []; 
    NL = len(channels)
    #print(NL)
    #lr = lr
    #NORMW = False
    pool = []; extra_pool = []

    for i in range(NL):

        if i ==0:
            net = Conv2d(3, channels[0], (kernel_size[0], kernel_size[0]), pad = pad[0], actp = power[i], batchnorm = Batchnorm, normdims = dims,
                         norm = norm, t = t, bias = bias, reth = reth, alpha=alpha[0], theta =theta[0],dropout = dropout[0], padding_mode = padding_mode)
            
        else:
            net = Conv2d(channels[i-1], channels[i], (kernel_size[i], kernel_size[i]), pad = pad[i], actp = power[i], batchnorm = Batchnorm, normdims = dims,
                         norm = norm, t = t, bias = bias, reth = reth, alpha=alpha[i], theta =theta[i],dropout = dropout[i], padding_mode = padding_mode)

        if i < freezelayer:
            net.load_state_dict(torch.load('./results/params_l' + str(i) +'_aug_lamda_new_tri_new_p1_all.pth'))
            #net.load_state_dict(torch.load('params_test.pth'))
            for param in net.parameters():
                param.requires_grad = False

        if NORMW:
            net.conv_layer.weight = torch.nn.Parameter(net.conv_layer.weight/ (net.conv_layer.weight.norm(p=2, dim=(-3, -2, -1), keepdim=True) + 1e-8))
        
        if pooltype[i] == 'Avg':
            pool.append(nn.AvgPool2d(kernel_size=pool_size[i], stride=stride_size[i], padding=padding[i], ceil_mode=True))
        else:
            pool.append(nn.MaxPool2d(kernel_size=pool_size[i], stride=stride_size[i], padding=padding[i], ceil_mode=True))

        extra_pool.append(nn.AvgPool2d(kernel_size=extra_pool_size[i], stride=extra_pool_size[i], padding=0, ceil_mode=True))
        net.to(device)
        nets.append(net)
        #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        #optimizer = Adam(net.parameters(), lr=lr[i])
        optimizer = AdamW(net.parameters(), lr=lr[i], weight_decay=weight_decay[i])

        optimizers.append(optimizer)

        #lr_lambda = create_lr_lambda(epochs, lr[i], 0, cutoffep)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = ExponentialLR(optimizer,gamma[i])
        
        schedulers.append(scheduler)

    #torch.manual_seed(1234)
    #current_rng_state = torch.get_rng_state()
    #current_rng_state2 = torch.get_rng_state()
    ##print(current_rng_state2[:10], len(current_rng_state2))  
    
    #print("Are current_rng_state and current_rng_state2 the same?", torch.equal(current_rng_state, current_rng_state2))
    #torch.set_rng_state(current_rng_state) 

    config = EvaluationConfig(device=device, dims=dims, dims_in=dims_in, dims_out=dims_out, stdnorm_out = stdnorm_out, 
                              out_dropout=out_dropout, Layer_out=Layer_out,pre_std = pre_std, all_neurons = all_neurons)

    if tr_and_eval:
        nets, _, testouts, _, _, all_pos, all_neg, _, tacc = train(
        nets, device, optimizers,schedulers, threshold1,threshold2, dims_in, dims_out, epochs, batchsize,pool,NORMW,a,b, lamda,lamda2,posact, cout,freezelayer
        ,period, extra_pool, tr_and_eval, Layer_out, all_neurons,trainloader, valloader, testloader, suptrloader,pre_std, triact,stdnorm_out,triact_pos, out_dropout,yita
        ,lambda_reg,lambda_covar, sup_gamma,sup_period,search, augment, Factor,p, lambda_weight, config)

    else:
        nets, trainouts, testouts, traintargets, testtargets, all_pos, all_neg, Dims = train(
            nets, device, optimizers,schedulers, threshold1,threshold2, dims_in, dims_out, epochs, batchsize,pool,NORMW,a,b, lamda,lamda2,posact, cout,freezelayer
            ,period, extra_pool, tr_and_eval, Layer_out, all_neurons,trainloader, valloader, testloader, suptrloader,pre_std, triact,stdnorm_out,triact_pos, out_dropout,yita
            ,lambda_reg,lambda_covar, sup_gamma,sup_period,search, augment, Factor,p, lambda_weight, config)
        #torch.manual_seed(42)
        tacc = evaluate_model(nets, pool, extra_pool, config, loaders, search, Dims) 
        #tacc = evaluation(trainouts, testouts, traintargets, testtargets, Layer_out)
        #tacc = evaluation_qudrant(trainouts, testouts, traintargets, testtargets, device, Layer_out)

    return tacc, all_pos, all_neg, testouts, nets


def main(lr, epochs, lamda, lamda2, lambda_reg, lambda_covar, cutoffep, device_num,tr_and_eval, th1, th2,period,gamma
         ,save_model, augment, Factor, weight_decay, yita, out_dropout, loaders, p, lambda_weight,tau,pool_type, seed_num):
    tacc, all_pos, all_neg, testouts, nets = hypersearch(
        channels = [96, 96*4, 96*16], 
        pad = [2, 1, 1],
        padding=[1,1,0],
        threshold1 = [th1, th1, th1], 
        threshold2 = [th2, th2, th2],
        lr = [lr, lr, lr],
        gamma = [gamma, gamma, gamma],
        dims =  (1,2,3),
        dims_in = (1,2,3), 
        dims_out = (1,2,3),
        power = [1,1,1], 
        Batchnorm = False, 
        epochs = epochs, 
        pool_size = [4, 4, 2], 
        batchsize = 100, 
        pooltype = ['Max','Max',pool_type], 
        kernel_size = [5, 3, 3],
        t = 1, 
        NORMW = False,  
        a = tau,
        b = tau,
        bias = True,
        reth = 0,
        lamda = [lamda, lamda, lamda],
        lamda2 = [lamda2, lamda2, lamda2],
        alpha = [0.0,0,0],
        theta = [-0.0,0,0],
        posact = False,
        cout = False,
        all_neurons = False,
        freezelayer = 2,
        Layer_out = [0,1,2],
        period = [period,period,period],
        extra_pool_size = [2, 2, 2],
        stride_size = [2,2, 2],
        tr_and_eval = tr_and_eval,
        dropout= [0.0, 0.0, 0.0],
        weight_decay=[weight_decay, weight_decay, weight_decay],
        pre_std = True,
        triact = False,
        stdnorm_out = True,
        norm = "stdnorm", #L2norm, stdnorm, no, 
        out_dropout = out_dropout,
        triact_pos = False,
        padding_mode = "reflect",
        yita = yita,
        lambda_reg = lambda_reg,
        sup_gamma = 0.7,
        sup_period = 4,
        lambda_covar = lambda_covar,
        search = False,
        cutoffep = cutoffep,
        device_num = device_num,
        augment = augment,
        Factor = Factor,
        loaders = loaders,
        p = p, 
        lambda_weight = lambda_weight,
        seed_num = seed_num)

    # save the model
    if save_model:
        for i, net in enumerate(nets):
            torch.save(net.state_dict(), './results/params_l'+str(i)+'_aug_lamda_new_cat_tri.pth')

    return tacc


def objective(trial):

    lr = trial.suggest_categorical('lr', [0.018, 0.02, 0.015, 0.01])
    lamda = trial.suggest_categorical('lamda', [0, 0.05,0.03, 0.02, 0.01, 0.04, 0.005, 0.001, 0.002]) 
    lamda2 = trial.suggest_categorical('lamda2', [0, 9, 10, 5, 8, 1, 2, 0.5, 0.1])
    th1 = trial.suggest_int('th1', 1, 9) 
    th2 = trial.suggest_int('th2', 1, 9) 
    lambda_covar = trial.suggest_categorical('lambda_covar', [0., 1, 2, 4, 5, 10, 0.5, 0.2, 0.1]) 
    gamma = trial.suggest_categorical('gamma', [ 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]) 
    yita = trial.suggest_categorical('yita', [1, 2])
    out_dropout = trial.suggest_categorical('out_dropout', [0])
    period = trial.suggest_categorical('period', [1000, 2000, 1500, 500])
    weight_decay = trial.suggest_categorical('weight_decay', [0, 1e-3, 1e-4, 3e-4]) 
    epochs = trial.suggest_int('epochs', 5, 40)
    tr_and_eval = False

    tsacc =  main(lr=lr, epochs=epochs, lamda=lamda, lamda2=lamda2, 
                lambda_reg=0, lambda_covar=lambda_covar, 
                cutoffep=5, device_num=1,tr_and_eval=tr_and_eval
                , th1=th1, th2=th2, period=period, gamma=gamma,
                save_model = False, augment = "no", Factor = 1, 
                weight_decay=weight_decay, yita=yita, out_dropout=out_dropout)
    
    if tr_and_eval:
        return 1- tsacc[-1][1]
    else:
        return 1- tsacc[1]



def create_objective(loaders):
    def objective(trial):
 
        #lr = trial.suggest_categorical('lr', [0.009, 0.01])
        lr = trial.suggest_float('lr',  0.0001, 0.001, step = 0.0001)
        #lamda = trial.suggest_categorical('lamda', [0]) 
        lamda = trial.suggest_float('lamda', 0.0, 0.002, step = 0.0001) 
        lamda2 = trial.suggest_categorical('lamda2', [0])
        #lamda2 = trial.suggest_float('lamda2', 0., 8, step = 0.1) 
        #th1 = trial.suggest_categorical('th1', [5]) 
        th1 = trial.suggest_int('th1', 0,10) 
        th2 = trial.suggest_int('th2', 0,10) 
        #th2 = trial.suggest_categorical('th2', [7]) 
        #lambda_covar = trial.suggest_float('lambda_covar', 0., 10, step = 0.1) 
        lambda_covar = trial.suggest_categorical('lambda_covar', [0]) 
        lambda_weight = trial.suggest_categorical('lambda_weight', [0])
        #lambda_weight = trial.suggest_float('lambda_weight', 0., 10) 
        gamma = trial.suggest_categorical('gamma', [1, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]) 
        #gamma = trial.suggest_categorical('gamma', [ 0.7])
        yita = trial.suggest_categorical('yita', [1])
        #out_dropout = trial.suggest_float('out_dropout', 0.1, 0.3, step = 0.1)
        out_dropout = trial.suggest_categorical('out_dropout', [0.1, 0.2])
        period = trial.suggest_categorical('period', [100, 200, 500, 1000, 2000, 1500])
        #period = trial.suggest_categorical('period', [1000])
        weight_decay = trial.suggest_categorical('weight_decay', [0, 1e-3, 1e-4, 3e-4]) 
        #weight_decay = trial.suggest_categorical('weight_decay', [1e-3]) 
        epochs = trial.suggest_int('epochs', 1, 30)
        #epochs = trial.suggest_categorical('epochs', [10]) 
        #p = trial.suggest_int('p', 1, 8)
        p = trial.suggest_categorical('p', [1]) 
        tr_and_eval = True
        #tau = trial.suggest_float('tau', 0.5, 1, step = 0.1)
        tau = trial.suggest_categorical('tau', [1])
        #loaders =  get_train(batchsize = 100, augment= "no", Factor= 1)
        #print('start')
        pool_type = trial.suggest_categorical('pool_type', ['Max','Avg']) 
        seed_num = trial.suggest_int('seed_num', 0,100000)

        tsacc =  main(lr=lr, epochs=epochs, lamda=lamda, lamda2=lamda2, 
                lambda_reg=0, lambda_covar=lambda_covar, 
                cutoffep=5, device_num=0,tr_and_eval=tr_and_eval
                , th1=th1, th2=th2, period=period, gamma=gamma,
                save_model = False, augment = "no", Factor = 1, 
                weight_decay=weight_decay, yita=yita, out_dropout=out_dropout, loaders=loaders
                , p = p, lambda_weight = lambda_weight,tau =tau,pool_type=pool_type, seed_num = seed_num)

        if tr_and_eval:
            return 1- tsacc[-1][1]
        else:
            return 1- tsacc[1]

    return objective

{'lr': 0.0002, 'lamda': 0.0005, 'lamda2': 0, 'th1': 1, 'th2': 10, 'lambda_covar': 0, 'lambda_weight': 0, 
 'gamma': 1, 'yita': 1, 'out_dropout': 0.2, 'period': 500, 'weight_decay': 0.0003, 'epochs': 20, 'p': 1, 
 'tau': 1, 'pool_type': 'Avg'}
{'lr': 0.0002, 'lamda': 0.0007, 'lamda2': 0, 'th1': 2, 'th2': 9, 'lambda_covar': 0, 'lambda_weight': 0, 
 'gamma': 1, 'yita': 1, 'out_dropout': 0.2, 'period': 500, 'weight_decay': 0, 'epochs': 15, 'p': 1, 'tau': 1, 
 'pool_type': 'Avg'}
{'lr': 0.0003, 'lamda': 0.001, 'lamda2': 0, 'th1': 3, 'th2': 9, 'lambda_covar': 0, 'lambda_weight': 0,
  'gamma': 1, 'yita': 1, 'out_dropout': 0.2, 'period': 2000, 'weight_decay': 0, 'epochs': 28, 'p': 1, 'tau': 1,
    'pool_type': 'Avg'}
if __name__ == "__main__":
    parser = argparse.ArgumentParser('ContrastFF script', parents=[get_arguments()])
    args = parser.parse_args()
    
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    
    #{'lr': 0.018, 'lamda': 0, 'lamda2': 0.644632082149124, 'th1': 3, 'th2': 5, 
    #'lambda_covar': 3.5385363579299187, 'lambda_weight': 5.315934820285312, 'gamma': 0.9, 
    #'yita': 1, 'out_dropout': 0, 'period': 2000, 'weight_decay': 0.0003, 'epochs': 19, 'p': 5}
    """
    loaders = get_train(batchsize=100, augment="no", Factor=1)

    tsacc =  main(lr=args.lr, epochs=args.epochs, lamda=args.lamda, lamda2=args.lamda2, 
     lambda_reg=args.lambda_reg, lambda_covar=args.lambda_covar, 
     cutoffep=args.cutoffep, device_num=args.device_num,tr_and_eval=args.tr_and_eval
     , th1=args.th1, th2=args.th2, period=args.period, gamma=args.gamma, augment=args.augment
     , weight_decay = args.weight_decay, yita = args.yita, Factor=args.factor, out_dropout=args.out_dropout
     , save_model = args.save_model, loaders=loaders, p = args.p, lambda_weight = args.lambda_weight, tau=args.tau,pool_type='Avg',
     seed_num=0)
    
     
    """
     # Define the range or list of values you want to loop over for each parameter
    search_space = {
    'lr': [0.0002],
    'lamda': [0.0005],
    'lamda2': [0],
    'th1': [6],
    'th2': [10],
    'lambda_covar': [0],
    'gamma': [1],
    'yita': [1],
    'out_dropout': [0.2],
    'period': [500],
    'weight_decay':[0.0003],
    'epochs': [25],
    'p': [1],
    'lambda_weight': [0],
    'pool_type': ['Avg'],
    'seed_num': [1234]
    }

    print('searching start...')
    #study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    #study.optimize(objective)
    loaders = get_train(batchsize=100, augment="no", Factor=1)
    objective_function = create_objective(loaders)
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    #study = optuna.create_study()
    #study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    study.optimize(objective_function)
    
    