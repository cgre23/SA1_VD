import os
import numpy as np
from matplotlib import pyplot as plt
#matplotlib graphs will be included in the notebook, next to the code:
import random
#add PyTorch and TorchVision (used for cropping etc.)
import torch
#import torchvision
import re
import json
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import pandas as pd
#import seaborn as sns
from skorch import NeuralNet

def train_NN(run_no):
    run='run'+run_no
    data = json.load( open( "results/"+run+"/prop_"+run+".json" ) )
    # Hyperparameters
    BATCH_SIZE = data['batch_size']
    MOMENTUM = data['momentum']
    k_folds = 4
    dataset_divider = 1
    INPUTS=len(data['features'])
    # mode 1 is random mode
    mode = 0
    n=1 # downsample

    class NN(nn.Module):
        def __init__(self, no_hidden_nodes, no_hidden_layers):
            super(NN, self).__init__()

            self.linear1 = nn.Linear(INPUTS, no_hidden_nodes)
            self.tanh1 = nn.Tanh()
            self.hidden_layers = nn.ModuleList()
            #self.hidden_layers_tanh = nn.ModuleList()
            for i in range(no_hidden_layers):
                self.hidden_layers.append(nn.Linear(no_hidden_nodes, no_hidden_nodes))
                self.hidden_layers.append(nn.Tanh())
            self.linear3 = nn.Linear(no_hidden_nodes, 2)

        def forward(self, x):
            x = self.linear1(x)
            x = self.tanh1(x)
            for hl in self.hidden_layers:
                x = hl(x)
            x = self.linear3(x)

            return x

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    cwd = os.getcwd()

    def train_model(model, epoch, train_loader, OPTIMIZER, fold):

        train_losses = []
        train_counter = []
        score = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #set network to training mode
        model.train()
        size = len(train_loader)*BATCH_SIZE
        last_r2 = 0
        stop_flag = False
        #for epoch in range(epochs):
            #iterate through data batches
        for batch_idx, (data, target) in enumerate(train_loader):
            tot_score = 0
            #reset gradients
            OPTIMIZER.zero_grad()

            #evaluate network with data
            output = model(data.to(device))

            #compute loss and derivative
            loss = F.mse_loss(output.to(device), target.to(device))
            loss.backward()

            #step optimizer
            OPTIMIZER.step()

            out=output.detach().cpu().numpy()
            targets=target.detach().cpu().numpy()
            r2 = pearsonr(targets.flatten(), out.flatten())[0]**2
            score.append(r2)

            #print out results and save to file
            if batch_idx % log_interval == 0:
                loss_n, current = loss.item(), batch_idx * len(data)
                logging.info(f"epoch: {epoch} loss: {loss_n:>5f} r2 {r2:>3f} [{current:>5d}/{size:>5d}]")

            tot_score = np.mean(score)
            train_losses.append(loss.item())
            #train_counter.append(
            #        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), cwd + f'/results/{run}/model-meas-fold-{run}.pth')
            torch.save(OPTIMIZER.state_dict(), cwd + f'/results/{run}/optimizer.pth')

        return train_losses, tot_score #train_counter

    def validation_model(model, valid_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        valid_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data.to(device))
                valid_loss += F.mse_loss(output.to(device), target.to(device), reduction='sum').item()
                #pred = output.data.max(1, keepdim=True)[1]
                #out=output.detach().cpu().numpy()
                #targets=target.detach().cpu().numpy()
                #r2 = pearsonr(targets.flatten(), out.flatten())[0]**2
                #score.append(r2)
        #total_score = np.mean(score)
        valid_loss /= len(valid_loader.dataset)
        #valid_losses.append(valid_loss)
        #print('\nValid set: Avg. loss: {:.4f}, R2 score: {:.4f}\n'.format(valid_loss, total_score))

        return valid_loss

    def reset_weights(m):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                #print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()



    log_interval = 1000
    prop={}
    run_6 = np.arange(1441752954, 1441769850)
    run_7 = np.arange(1441789915, 1441796858)
    run_9 = np.arange(1441821988, 1441839339)
    run_10 = np.arange(1441839422, 1441853277)
    run_11 = np.arange(1441853377, 1441871154)
    run_12 = np.arange(1441871620, 1441877857)
    run_13 = np.arange(1441886045, 1441904537)
    run_14 = np.arange(1441904651, 1441910194)
    run_15 = np.arange(1441914642, 1441925000)

    run710 = np.hstack([run_7, run_10])

    #run_ = run_6
    #run_length = int(len(run_))
    #run_ = np.random.choice(run_, int(run_length/dataset_divider))
    if run_no == '6':
        run_ = run_6
    elif run_no == '7':
        run_ = run_7
    elif run_no == '9':
        run_ = run_9
    elif run_no == '10':
        run_ = run_10
    elif run_no == '11':
        run_ = run_11
    elif run_no == '12':
        run_ = run_12
    elif run_no == '13':
        run_ = run_10
    elif run_no == '14':
        run_ = run_11
    elif run_no == '15':
        run_ = run_12

    df = pd.read_hdf("data/merged.h5", key='df')
    df = df.filter(items=run_, axis=0)
    df = df.drop('date', axis=1)
    df = df.iloc[::n, :]
    #df.fillna(df.mean(), inplace=True)
    SA1_BPMs_x = []
    SA1_BPMs_y = []
    SA1_XGMs = []
    channels = []
    if mode == 1:
        p = re.compile('^(/XFEL\.DIAG/BPM/BPME\.22[4-9]..SA1/X\.TD|/XFEL\.DIAG/BPM/BPME\.23...SA1/X\.TD|/XFEL\.DIAG/BPM/BPME\.24[0-6]..SA1/X\.TD)')
        q = re.compile('^(/XFEL\.DIAG/BPM/BPME\.22[4-9]..SA1/Y\.TD|/XFEL\.DIAG/BPM/BPME\.23...SA1/Y\.TD|/XFEL\.DIAG/BPM/BPME\.24[0-6]..SA1/Y\.TD)')
    else:
        p = re.compile('^(/XFEL\.DIAG/BPM/BPME\.22[5-9]..SA1/X\.TD|/XFEL\.DIAG/BPM/BPME\.23[0-3]..SA1/X\.TD)')
        q = re.compile('^(/XFEL\.DIAG/BPM/BPME\.22[5-9]..SA1/Y\.TD|/XFEL\.DIAG/BPM/BPME\.23[0-3]..SA1/Y\.TD)')

    r = re.compile('^(/XFEL\.FEL/XGM/XGM\.26...T.*/INTENSITY\.TD)')
    SA1_BPMs_x = [ s for s in list(data['features']) if p.match(s) ]
    SA1_BPMs_y = [ s for s in list(data['features']) if q.match(s) ]
    SA1_XGMs = [ s for s in list(data['features']) if r.match(s) ]
    XGM_labels = [ re.findall(r'\d+\d+\d+\d+', l)[0] for l in SA1_XGMs ]
    BPM_labels = [ re.findall(r'\d+\d+\d+\d+', l)[0] for l in SA1_BPMs_x ]


    features = data['features']
    targets = data['targets']
    inputs_outputs = features+targets
    #traindf, testdf = np.split(df[inputs_outputs].sample(frac=1, random_state=42), 
    #                       [int(.7*len(df))])
    traindf, validdf = np.split(df[inputs_outputs], 
                           [int(.8*len(df))])

    data['norm_min'] = traindf.min().tolist()
    data['norm_max'] = traindf.max().tolist()
    data['inputs_outputs'] = traindf.columns.tolist()

    result = {}
    # Define the K-fold Cross Validator
    #kfold = KFold(n_splits=k_folds, shuffle=True)
    #norm_df=(traindf-traindf.mean())/traindf.std()
    norm_df=(traindf-traindf.min())/(traindf.max()-traindf.min())
    normvalid_df=(validdf-traindf.min())/(traindf.max()-traindf.min())
    norm_df=norm_df.astype(float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    LEARNING_RATE = data['learning_rate']
    EPOCHS = data['epochs']
    HIDDEN_LAYERS = data['hidden_layers']
    HIDDEN_NODES = data['hidden_nodes']

    #MOMENTUM = gs.best_params_['optimizer__momentum'] # 0.4
    logging.info('Training NN: Learning rate: %s Epochs: %s Hidden layers: %s Hidden nodes: %s' % (LEARNING_RATE, EPOCHS, HIDDEN_LAYERS, HIDDEN_NODES))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(HIDDEN_NODES, HIDDEN_LAYERS)
    model = model.to(device)
    OPTIMIZER = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                              momentum=MOMENTUM)

    traindataset = torch.utils.data.TensorDataset(torch.tensor(norm_df[features].values.astype(np.float32)), torch.tensor(norm_df[targets].values.astype(np.float32)))
    validdataset = torch.utils.data.TensorDataset(torch.tensor(normvalid_df[features].values.astype(np.float32)), torch.tensor(normvalid_df[targets].values.astype(np.float32)))
    #datasets = train_val_dataset(dataset)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(validdataset, batch_size=BATCH_SIZE)


    # Early stopping
    last_loss = 100
    patience = 30
    triggertimes = 0
    
    model.apply(reset_weights)

    for t in range(int(EPOCHS)*4):
        epoch = t+1
        fold = 10
        train_loss, tot_score = train_model(model, epoch, train_loader, OPTIMIZER, fold)

        current_loss = validation_model(model, valid_loader)

        if current_loss > last_loss:
            trigger_times += 1
            logging.info('Trigger Times: %s' % str(trigger_times))

            if trigger_times >= patience:
                logging.info('Early stopping!')
                break

        else:
            trigger_times = 0

        last_loss = current_loss

        # Process is complete.
    logging.info('Training process has finished. Saving trained model.\nStart to test process.')
    json.dump( data, open( "results/"+run+"/prop_"+run+".json", 'w' ) )

    test_losses, score = [], []
    total_score = 0
    model.eval()
    test_loss, correct = 0, 0
    output_l = []
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data.to(device))
            test_loss += F.mse_loss(output.to(device), target.to(device), reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            out=output.detach().cpu().numpy()
            output_l.append(out)
            targets=target.detach().cpu().numpy()
            r2 = pearsonr(targets.flatten(), out.flatten())[0]**2
            score.append(r2)
    total_score = np.mean(score)
    logging.info('Evaluation with validation dataset: %s' % str(total_score))
    return total_score



