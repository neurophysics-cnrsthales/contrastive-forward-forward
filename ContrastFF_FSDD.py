import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, StepLR, LinearLR
from torch.utils.data import Dataset, DataLoader
from dataset import SpokenDigitDataset, collate
import torch.nn.functional as F

import importlib
import argparse
#importlib.reload(tools)
import optuna




def stdnorm (x, dims = [1,2,3]):

    x = x - torch.mean(x, dim=(dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(dims), keepdim=True))

    return x

def l2norm (x, dims = [1,2,3]):

    #x = x - torch.mean(x, dim=(dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(dims), keepdim=True))
    x = x / (x.norm(p=2, dim=(dims), keepdim=True) + 1e-10)
    return x

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

class BiRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, device, stop_grad = False, nonlinearity = 'tanh'
                 , norm_h = "L2norm", norm_in = "no", norm_out = "std"):
        super(BiRNN, self).__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.rnn_f = nn.RNN(embedding_dim, hidden_dim, batch_first=True, nonlinearity=nonlinearity) #'tanh' or 'relu'
        self.rnn_b = nn.RNN(embedding_dim, hidden_dim, batch_first=True, nonlinearity=nonlinearity)
        self.stop_grad = stop_grad
        #self.fc1 = nn.Linear(hidden_dim*2, 8)  # *2 because it's bidirectional
        #self.fc2 = nn.Linear(8, output_dim)
        #self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        self.device = device
        if norm_h == "L2":
            self.norm = L2norm(dims = -1)
        elif norm_h == "std":
            self.norm_h = standardnorm(dims = -1)
        else:
            self.norm_h = nn.Identity()
        
        if norm_in == "std":
            self.norm_in = standardnorm(dims = -1)
        elif norm_in == "L2":
            self.norm_in = L2norm(dims = -1)
        else:
            self.norm_in = nn.Identity()

        if norm_out == "std":
            self.norm_out = standardnorm(dims = -1)
        elif norm_in == "L2":
            self.norm_out = L2norm(dims = -1)
        else:
            self.norm_out = nn.Identity()

    def forward(self, x):
        """
        # x:        [batch_size, len_seq]
        # embedded: [batch_size, len_seq, feature_dim]
        # hidden_forward and hidden_backward: [1, batch_size, hidden_dim]
        # hiddens_forward: [batch_size, len_seq, hidden_dim]
        """
        if x.size(2) > self.embedding_dim:
            embedded = x[:,:, :x.size(2)//2]+  x[:,:, x.size(2)//2:]
            embedded_reversed = torch.flip(embedded, [1])
        else:
            embedded = x 
            embedded_reversed = torch.flip(embedded, [1])

        # Normalize or std the inputs:
        x = self.norm_in(x)
        #rnn_out, hidden = self.rnn(embedded)
        hidden_forward = torch.zeros(1, embedded.size(0), self.hidden_dim, device=self.device)
        hidden_backward = torch.zeros(1, embedded.size(0), self.hidden_dim, device=self.device)

        hiddens_forward = []
        hiddens_backward = []

        for t in range(embedded.size(1)):
            # Forward RNN
            _, hidden_forward_new = self.rnn_f(embedded[:, t:t+1, :], hidden_forward)
            #hidden_forward = (hidden_forward[0].detach(), hidden_forward[1].detach())
            #hidden_forward = hidden_forward_new.detach()
            #hidden_forward = hidden_forward_new
            # Backward RNN
            _, hidden_backward_new = self.rnn_b(embedded_reversed[:, t:t+1, :], hidden_backward)
            #hidden_backward = hidden_backward_new.detach()
            """
            if self.stop_grad:
                hidden_forward = hidden_forward_new.detach()
                hidden_backward = hidden_backward_new.detach()
            else:
                hidden_forward = hidden_forward_new
                hidden_backward = hidden_backward_new
            """
            if self.stop_grad:
                hidden_forward =  self.norm_h(hidden_forward_new).detach()
                hidden_backward = self.norm_h(hidden_backward_new).detach()
            else:
                hidden_forward = self.norm_h(hidden_forward_new)
                hidden_backward = self.norm_h(hidden_backward_new)

            hiddens_forward.append(hidden_forward_new)
            hiddens_backward.append(hidden_backward_new)
            
            #hidden_backward = (hidden_backward[0].detach(), hidden_backward[1].detach())
        # Assuming the output of LSTM is only needed from the final time step
        #print(lstm_out.shape, hidden.shape)
        #hiddens = lstm_out[:,-1,:]
        hiddens_forward = torch.stack(hiddens_forward, dim=0).squeeze(1).transpose(0,1)
        hiddens_backward = torch.stack(hiddens_backward, dim=0).squeeze(1).transpose(0,1).flip([1])

        #print(hiddens_forward.shape, hiddens_backward.shape)
        hiddens_last = torch.cat((hidden_forward[0], hidden_backward[0]), dim = -1)
        #dense_outputs = self.relu(self.fc1(hidden))
        #outputs = self.fc2(dense_outputs)
        return hiddens_last, torch.cat((hiddens_forward, hiddens_backward), dim = -1)


class Readout(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Readout, self).__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim*2, output_dim)  # *2 because it's bidirectional
        #self.fc2 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden):
        #embedded = self.embedding(x)
        #lstm_out, (hidden, cell) = self.lstm(embedded)
        # Assuming the output of LSTM is only needed from the final time stepf
        #hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        outputs = self.fc1(hidden)
        #outputs =(self.fc2(dense_outputs))
        return outputs

def get_pos_neg_batch_imgcats_sup0(x1, x2, targets, num_classes, p = 1):
    # group the data with the same targets together into lists
    """ 
    x1 and x2: [batch_size, seq_len]
    """
    Batch_lists1 = [] # 10 lists that group togher the same class
    Batch_lists2 = []

    for i in range(0, num_classes):
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
        # random_indices = (torch.randperm(len(batch1)))
        #batch_pos = torch.cat((batch1, batch2[random_indices]), dim = 1)
        batch_pos = torch.cat((batch1, batch2), dim = 2)
        #print(len(batch1))
        

        shifted_idx = classes.roll(shifts=-i)[1:]
        selected_tensors = [filtered_batch_list2[idx] for idx in shifted_idx]
        # Concatenate the selected tensors
        neg_all = torch.cat(selected_tensors)
        #neg_all = torch.cat(filtered_batch_list2[classes.roll(shifts=-i)[1:].long()])
        #random_ids = torch.randint(0, len(neg_all), (len(batch1),))
        #batch_neg = torch.cat((batch1, neg_all[random_ids]), dim = 1)

        random_ids = torch.randint(0, len(neg_all), (len(batch1), p))
        batch_neg = torch.cat([torch.cat((batch1, neg_all[random_ids[:, i]]), dim=2) for i in range(p)])
        

        batch_poses.append(batch_pos)
        batch_negs.append(batch_neg)

    return torch.cat(batch_poses), torch.cat(batch_negs)

def get_pos_neg_batch_imgcats(batch_pos1, batch_pos2, p = 2):

    batch_size = len(batch_pos1)

    batch_pos =torch.cat((batch_pos1, batch_pos2), dim = -1)

    #create negative samples
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p,batch_size - 1)]
    labeles = torch.arange(batch_size)

    batch_negs = []
    for i in random_indices:
        batch_neg = batch_pos2[(labeles+i)%batch_size]
        batch_neg = torch.cat((batch_pos1, batch_neg), dim = -1)
        batch_negs.append(batch_neg)
    
    return batch_pos, torch.cat(batch_negs)

def contains_nan(model):
    for p in model.parameters():
        if torch.isnan(p).any():
            return True
    return False

def train(model,sup_train_loader, test_loader, val_loader, out_dropout, hidden_dim, output_dim, optimizer
,scheduler, threshold1, threshold2, tau, lamda, epochs, train_loader,p, device,tr_and_eval,clr):

    #loss_function = nn.CrossEntropyLoss()  # This automatically applies Softmax for you
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train without training the LSTM layer, only Readout layer is being trained
    # Assuming X_train and y_train are already tensors


    #threshold1 = 2
    #threshold2 = 2
    #tau = 0.8
    #lamda = 0.003
    #scheduler = ExponentialLR(optimizer, 0.5)

    model.train()
    #yita = 1

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        #total_predictions = 0
        #correct_predictions = 0
        goodness_pos, goodness_neg = 0, 0
        for inputs,_, labels in train_loader:
            inputs = inputs.to(device)
            #x_pos, x_neg = get_pos_neg_batch_imgcats_sup0(inputs, inputs, labels, 10, p = p)
            x_pos, x_neg = get_pos_neg_batch_imgcats(inputs, inputs, p = p)
            optimizer.zero_grad()
            _, hiddens_all_pos = model(x_pos)
            _, hiddens_all_neg = model(x_neg)
            #hiddens = model(inputs);outputs = readout(hiddens)
            #loss = loss_function(outputs, labels)
            yforgrad = hiddens_all_pos.pow(2).mean([-1])

            loss =  torch.log(1 + torch.exp(
                            tau*(- yforgrad  + threshold1))).mean([1]).mean(
                            ) + torch.log(1 + torch.exp(
                                tau*(hiddens_all_neg.pow(2).mean([-1])  - threshold2))).mean([1]).mean() + lamda * torch.norm(yforgrad, p=2, dim = (1)).mean(
                                ) #+ lamda2*(F.relu(yita - yfor_reg.std(dim = 0)).mean())
            loss.backward()
            optimizer.step()
            
            good_pos = hiddens_all_pos.pow(2).mean([-1]).mean(1).mean().item()
            good_neg = hiddens_all_neg.pow(2).mean([-1]).mean(1).mean().item()
            #print(i, "pos: ", good_pos)
            #print(i, "neg: ", good_neg)
            running_loss += loss.item()
            goodness_pos += good_pos
            goodness_neg += good_neg
        
        if contains_nan(model):
            raise optuna.TrialPruned()
        
        scheduler.step()

        print(f'epoch {epoch+1} learning rate: {scheduler.get_last_lr()[0]}') 
        print("mean goodness for pos: ", (goodness_pos)/len(train_loader))
        print("mean goodness for neg: ", (goodness_neg)/len(train_loader))

        if tr_and_eval:
            if epoch>1 and epoch<(epochs-1) and epoch%1==0:
                acc = evaluate(model,sup_train_loader, train_loader, test_loader, val_loader
                , out_dropout, hidden_dim, output_dim, device,clr)

    return model

def evaluate(model,sup_train_loader, train_loader,test_loader, val_loader, out_dropout, hidden_dim, output_dim, device,clr):

    current_rng_state = torch.get_rng_state()

    torch.manual_seed(42)
    readout = nn.Sequential(
        nn.Dropout(out_dropout),  # Dropout layer with 50% drop probability
        nn.Linear(hidden_dim*2, output_dim)  # 
    )

    readout = readout.to(device)

    loss_function = nn.CrossEntropyLoss()  # This automatically applies Softmax for you
    optimizer = torch.optim.Adam(readout.parameters(), lr=clr)
    nb_epochs = 10
    sup_lr_scheduler = CustomStepLR(optimizer, nb_epochs=nb_epochs)
    # Train without training the LSTM layer, only Readout layer is being trained

    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        readout.train()
        running_loss = 0.0
        total_predictions = 0
        correct_predictions = 0
        for inputs, _, labels in sup_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            x_pos = torch.cat((inputs,inputs), dim = -1)
            optimizer.zero_grad()
            with torch.no_grad():
                hiddens_last_pos, _  = model(x_pos)

            outputs = model.norm_out(hiddens_last_pos)
            outputs = readout(outputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, -1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        sup_lr_scheduler.step()
        train_accuracy = correct_predictions / total_predictions

        if epoch % 2 == 0 or epoch == (nb_epochs-1):
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Training Accuracy: {train_accuracy}")
            # Validation loss
            readout.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            with torch.no_grad():
                for inputs,_, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    x_pos = torch.cat((inputs,inputs), dim = -1)
                    hiddens_last_pos, _  = model(x_pos)

                    outputs = model.norm_out(hiddens_last_pos)
                    outputs = readout(outputs)
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

            val_accuracy = correct_predictions / total_predictions
            print(f"Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}")

    # test
    readout.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs,_, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            x_pos = torch.cat((inputs,inputs), dim = -1)
            hiddens_last_pos, _  = model(x_pos)

            outputs = model.norm_out(hiddens_last_pos)
            outputs = readout(outputs)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    test_accuracy = correct_predictions / total_predictions
    print(f"Test Loss: {test_loss/len(test_loader)}, Test Accuracy: {test_accuracy}")

    torch.set_rng_state(current_rng_state)

    return [train_accuracy, (test_accuracy+val_accuracy)/2]

def get_train(batch_size):
    #batch_size = 64
    torch.manual_seed(1234)
    dataset_path = './data/dataset'
    sampling_rate = 16000
    n_mfcc = 39
    dataset = SpokenDigitDataset(dataset_path, sampling_rate, n_mfcc)
    train_valid_test_split = [80, 10, 10]
    train_dataset, valid_dataset, test_dataset = dataset.split_dataset(train_valid_test_split)
    sup_train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate, shuffle=False)

    return sup_train_loader, train_loader, test_loader, val_loader

def hypersearch(threshold1, threshold2, tau, lamda, epochs, lr, weight_decay, gamma, out_dropout, p, loaders,
                nonlinearity, norm_h, norm_in, norm_out, device,test,seed_num,tr_and_eval,clr):

    feature_size = 39
    hidden_dim = 500
    output_dim = 10

    #batch_size = 32

    #dataset = tools.TimitDataset(batch_size, data_path='../data/TIMIT_processed', preproc='mfccs', use_reduced_phonem_set=True)

    #train_loader, test_loader, val_loader = get_train(dataset, batch_size)
    sup_train_loader, train_loader, test_loader, val_loader = loaders

    #torch.manual_seed(1234)
    torch.manual_seed(seed_num)

    if test:
        val_loader = test_loader
    #else:
        #ts_loader = val_loader
    
    model = BiRNN(feature_size, hidden_dim, device, nonlinearity = nonlinearity
                 , norm_h = norm_h, norm_in = norm_in, norm_out = norm_out, stop_grad = True).to(device)
    #readout = Readout(hidden_dim, output_dim)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma)
    model = train(model, sup_train_loader, test_loader, val_loader, out_dropout, hidden_dim
    , output_dim, optimizer, scheduler, threshold1, threshold2, tau, lamda, epochs, train_loader, p, device,tr_and_eval,clr)
    
    
    acc = evaluate(model, sup_train_loader, train_loader, test_loader, val_loader, out_dropout, hidden_dim, output_dim, device,clr)

    return acc, model

def main(threshold1, threshold2, tau, lamda, epochs, lr, weight_decay, gamma, p, loaders,nonlinearity, norm_h, norm_in, norm_out
         ,device,test,seed_num,tr_and_eval,clr):

    acc,model = hypersearch(
        threshold1 = threshold1, 
        threshold2 = threshold2, 
        tau = tau, 
        lamda = lamda, 
        epochs = epochs, 
        lr = lr, 
        weight_decay = weight_decay, 
        gamma = gamma,
        out_dropout = 0,
        p = p,
        loaders = loaders,
        nonlinearity = nonlinearity, 
        norm_h = norm_h, 
        norm_in = norm_in, 
        norm_out = norm_out, 
        device = device,
        test = test,
        seed_num = seed_num,
        tr_and_eval = tr_and_eval,
        clr = clr)
    
    # save the model
    
    return acc



def create_objective(loaders, device):
    def objective(trial):

        lr = trial.suggest_categorical('lr', [0.001, 0.0001, 1e-5,  2e-5, 5e-5,8e-5, 2e-4, 5e-4,6e-4, 8e-4, 2e-3,5e-3,1e-2])

        lamda = trial.suggest_float('lamda', 0.00, 0.01, step = 0.0001) 
        
        th1 = trial.suggest_float('th1', 0, 10, step = 1) 
        th2 = trial.suggest_float('th2', 0, 10, step = 1) 
        
        gamma = trial.suggest_categorical('gamma', [1, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]) 
        
        weight_decay = trial.suggest_categorical('weight_decay', [0, 1e-3, 1e-4, 3e-4]) 
        epochs = trial.suggest_int('epochs', 1, 50)
        p = trial.suggest_categorical('p', [1]) 
        tr_and_eval = True
        #tau = trial.suggest_float('tau', 0.4, 1, step = 0.1)
        tau = trial.suggest_categorical('tau', [1.0])
        nonlinearity = trial.suggest_categorical('nonlinearity', ['relu'])
        #nonlinearity = trial.suggest_categorical('nonlinearity', ['tanh', 'relu'])
        norm_h = trial.suggest_categorical('norm_h', ["std"])
        norm_in = trial.suggest_categorical('norm_in', ["no"])
        norm_out = trial.suggest_categorical('norm_out', ["L2norm"])
        seed_num = trial.suggest_int('seed_num', 0,100000)
        clr = trial.suggest_float('clr', 0.00, 0.01, step = 0.0001) 

        tsacc =  main(lr=lr, epochs=epochs, lamda=lamda, threshold1=th1, threshold2=th2, gamma=gamma,
                weight_decay=weight_decay, loaders=loaders
                , p = p,tau =tau, nonlinearity = nonlinearity, 
                norm_h = norm_h, norm_in = norm_in, norm_out = norm_out, device = device
                , test=True, seed_num = seed_num,tr_and_eval = tr_and_eval,clr = clr)
        
        return 1- tsacc[1]

    return objective


def get_arguments(): 
    #lr, epochs, lamda, lamda2, lambda_reg, lambda_covar, cutoffep, device_num

    parser = argparse.ArgumentParser(description="Pretrain a RNN using contrastiveFF", add_help=False)


    # Optim
    parser.add_argument("--epochs", type=int, default=5,
                        help='Number of epochs')
    parser.add_argument("--lr", type=float, default=0.01,
                        help='Base learning rate')
    parser.add_argument("--gamma", type=float, default=0.8,
                        help='exponential decay rate')
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help='weight_decay rate')
   
    
    # Loss
    parser.add_argument("--th1", type=int, default=2,
                        help='thre1 for positive samples')
    parser.add_argument("--th2", type=int, default=2,
                        help='thre2 for negative samples')
    parser.add_argument("--lamda", type=float, default=0.003,
                        help='L2 norm regularization loss coefficient')
    parser.add_argument("--p", type=int, default=3,
                        help='Number of negative samples for each postive')
    parser.add_argument("--tau", type=float, default=0.8,
                        help='temperature')
    parser.add_argument("--enable_gpu", action='store_true',
                        help='train with gpu')
    parser.add_argument('--device_num',type=int, default=0,
                        help='device to use for training / testing')
    parser.add_argument("--test", action='store_true',
                        help='use testset')

    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('ContrastFF TIMIT script', parents=[get_arguments()])
    args = parser.parse_args()
    print('searching start...')
    #study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    #study.optimize(objective)
    loaders = get_train(64)
    # Automatically select GPU if available, else fall back to CPU

    if torch.cuda.is_available() and args.enable_gpu:
        device = 'cuda:' + str(args.device_num) 
    else:
        device = 'cpu'
    
    print(f'Using device: {device}')
    search_space = {
    'lr': [2e-5],
    'clr': [5e-4],
    'lamda': [0.0075],
    'th1': [0],
    'th2': [1],
    'gamma': [0.7],
    'weight_decay':[0],
    'epochs': [10],
    'p': [1],
    'seed_num': [1234,10,100,1000]
    }

    objective_function = create_objective(loaders, device)
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    #study = optuna.create_study()
    #study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    study.optimize(objective_function)
    