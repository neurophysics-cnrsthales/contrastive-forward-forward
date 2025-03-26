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
    """
    A Bidirectional Recurrent Neural Network (BiRNN) using two RNN layers (forward & backward).

    This network processes sequential input using:
    - A forward RNN (`rnn_f`) to capture past dependencies.
    - A backward RNN (`rnn_b`) to capture future dependencies.
    - Optional normalization for hidden states, inputs, and outputs.

    Args:
        embedding_dim (int): Dimension of input features.
        hidden_dim (int): Number of hidden units in each RNN.
        device (str): Device to run the model ('cpu' or 'cuda').
        stop_grad (bool, optional): If True, stops gradient flow after each timestep (default: False).
        nonlinearity (str, optional): Nonlinearity for RNN ('tanh' or 'relu') (default: 'tanh').
        norm_h (str, optional): Type of normalization for hidden states ('L2norm', 'std', or 'no') (default: 'L2norm').
        norm_in (str, optional): Type of normalization for inputs ('L2norm', 'std', or 'no') (default: 'no').
        norm_out (str, optional): Type of normalization for outputs ('L2norm', 'std', or 'no') (default: 'std').
    """
    def __init__(self, embedding_dim, hidden_dim, device, stop_grad = False, nonlinearity = 'tanh'
                 , norm_h = "L2norm", norm_in = "no", norm_out = "std"):
        super(BiRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.rnn_f = nn.RNN(embedding_dim, hidden_dim, batch_first=True, nonlinearity=nonlinearity) #'tanh' or 'relu'
        self.rnn_b = nn.RNN(embedding_dim, hidden_dim, batch_first=True, nonlinearity=nonlinearity)
        self.stop_grad = stop_grad
        
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
        Forward pass of the BiRNN.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_length, feature_dim]

        Returns:
            hiddens_last (Tensor): Concatenated last hidden states from forward and backward RNNs.
                                   Shape: [batch_size, hidden_dim * 2]
            hidden_states (Tensor): Concatenated sequence of forward and backward hidden states.
                                    Shape: [batch_size, seq_length, hidden_dim * 2]
        """
        """
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
            # Backward RNN
            _, hidden_backward_new = self.rnn_b(embedded_reversed[:, t:t+1, :], hidden_backward)
            
            if self.stop_grad:
                hidden_forward =  self.norm_h(hidden_forward_new).detach()
                hidden_backward = self.norm_h(hidden_backward_new).detach()
            else:
                hidden_forward = self.norm_h(hidden_forward_new)
                hidden_backward = self.norm_h(hidden_backward_new)

            hiddens_forward.append(hidden_forward_new)
            hiddens_backward.append(hidden_backward_new)
            

        hiddens_forward = torch.stack(hiddens_forward, dim=0).squeeze(1).transpose(0,1)
        hiddens_backward = torch.stack(hiddens_backward, dim=0).squeeze(1).transpose(0,1).flip([1])

        hiddens_last = torch.cat((hidden_forward[0], hidden_backward[0]), dim = -1)
        return hiddens_last, torch.cat((hiddens_forward, hiddens_backward), dim = -1)


class Readout(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Readout, self).__init__()
        self.fc1 = nn.Linear(hidden_dim*2, output_dim)  # *2 because it's bidirectional
        self.relu = nn.ReLU()

    def forward(self, hidden):
        outputs = self.fc1(hidden)
        return outputs

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

    batch_pos =torch.cat((batch_pos1, batch_pos2), dim = -1)
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p,batch_size - 1)]
    labeles = torch.arange(batch_size)

    batch_negs = []
    for i in random_indices:
        batch_neg = batch_pos2[(labeles+i)%batch_size]
        batch_neg = torch.cat((batch_pos1, batch_neg), dim = -1)
        batch_negs.append(batch_neg)
    
    return batch_pos, torch.cat(batch_negs)



def train(model,sup_train_loader, test_loader, val_loader, out_dropout, hidden_dim, output_dim, optimizer
    ,scheduler, threshold1, threshold2, tau, lamda, epochs, train_loader,p, device,tr_and_eval,clr):

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        goodness_pos, goodness_neg = 0, 0
        for inputs,_, labels in train_loader:
            inputs = inputs.to(device)
            x_pos, x_neg = get_pos_neg_batch_imgcats(inputs, inputs, p = p)
            optimizer.zero_grad()
            _, hiddens_all_pos = model(x_pos)
            _, hiddens_all_neg = model(x_neg)

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
            running_loss += loss.item()
            goodness_pos += good_pos
            goodness_neg += good_neg
        
       
        scheduler.step()

        print(f'epoch {epoch+1} learning rate: {scheduler.get_last_lr()[0]}') 
        print("mean goodness for pos: ", (goodness_pos)/len(train_loader))
        print("mean goodness for neg: ", (goodness_neg)/len(train_loader))

        if tr_and_eval:
            if epoch>1 and epoch<(epochs-1) and epoch%1==0:
                acc = evaluate(model,sup_train_loader,  test_loader, val_loader
                , out_dropout, hidden_dim, output_dim, device,clr)

    return model

def evaluate(model,sup_train_loader, test_loader, val_loader, out_dropout, hidden_dim, output_dim, device,clr):

    """
    Evaluates the trained model on training, validation, and test datasets.

    Args:
        model (nn.Module): Trained BiRNN model.
        sup_train_loader (DataLoader): DataLoader for supervised training.
        test_loader (DataLoader): DataLoader for test dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
        out_dropout (float): Dropout rate for the classifier.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Number of output classes.
        device (str): Computation device ('cpu' or 'cuda').
        clr (float): Initial Learning rate of the classifier.

    Returns:
        list: [train accuracy, averaged test/validation accuracy].
    """

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
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(sup_train_loader)}, Training Accuracy: {train_accuracy}")
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
     """
    Loads the SpokenDigitDataset and splits it into training, validation, and test sets.

    Args:
        batch_size (int): Batch size for training.

    Returns:
        tuple: DataLoaders for supervised training, training, validation, and testing.
    """
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
    """
    Conducts training and evaluation of the BiRNN model.

    Args:
        threshold1 (float): Threshold values for Positive examples.
        threshold2 (float): Threshold values for Negative examples.
        tau (float): Temperature, default 1.
        lamda (float): L2 regularization factor.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay factor.
        gamma (float): Learning rate decay factor.
        out_dropout (float): Dropout rate for output.
        p (int): Number of negative samples per positive, default=1.
        loaders (tuple): DataLoaders for training, validation, and testing.
        nonlinearity (str): Activation function.
        norm_h (str): Normalization type for hidden layers.
        norm_in (str): Normalization type for input.
        norm_out (str): Normalization type for output.
        device (str): Computation device ('cpu' or 'cuda').
        test (bool): Whether to evaluate on test set.
        seed_num (int): Random seed for reproducibility.
        tr_and_eval (bool): Whether to train and evaluate together.
        clr (float): Initial learning rate of the classifier

    Returns:
        tuple: Model accuracy and trained model.
    """
    feature_size = 39
    hidden_dim = 500
    output_dim = 10

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
    
    
    acc = evaluate(model, sup_train_loader, test_loader, val_loader, out_dropout, hidden_dim, output_dim, device,clr)

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



def get_arguments():
    """
    Parses command-line arguments for training the model.

    Returns:
        argparse.ArgumentParser: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Pretrain a BiRNN using SCFF", add_help=False)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Base learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, help="Exponential decay rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay rate")
    
    # Loss parameters
    parser.add_argument("--th1", type=int, default=0, help="Threshold for positive samples")
    parser.add_argument("--th2", type=int, default=1, help="Threshold for negative samples")
    parser.add_argument("--lamda", type=float, default=0.0075, help="L2 norm regularization coefficient")
    parser.add_argument("--p", type=int, default=1, help="Number of negative samples for each positive sample")
    parser.add_argument("--tau", type=float, default=1, help="Temperature parameter")

    # Device settings
    parser.add_argument("--enable_gpu", action="store_true", help="Enable GPU training if available")
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument("--test", action="store_true", help="Use test set for evaluation")

    # Additional parameters
    parser.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "tanh"], help="Activation function")
    parser.add_argument("--norm_h", type=str, default="std", help="Normalization for hidden layers")
    parser.add_argument("--norm_in", type=str, default="no", help="Normalization for input")
    parser.add_argument("--norm_out", type=str, default="L2norm", help="Normalization for output")
    parser.add_argument("--seed_num", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--tr_and_eval", action="store_true", help="Train and evaluate together")
    parser.add_argument("--clr", type=float, default=5e-4, help="Initial learning rate of the classifier")

    return parser


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser("SCFF TIMIT script", parents=[get_arguments()])
    args = parser.parse_args()

    # Load training data
    loaders = get_train(64)

    # Automatically select device
    if torch.cuda.is_available() and args.enable_gpu:
        device = f'cuda:{args.device_num}'
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Run main function with parsed arguments
    tsacc = main(
        threshold1=args.th1,
        threshold2=args.th2,
        tau=args.tau,
        lamda=args.lamda,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        p=args.p,
        loaders=loaders,
        nonlinearity=args.nonlinearity,
        norm_h=args.norm_h,
        norm_in=args.norm_in,
        norm_out=args.norm_out,
        device=device,
        test=args.test,
        seed_num=args.seed_num,
        tr_and_eval=args.tr_and_eval,
        clr=args.clr
    )