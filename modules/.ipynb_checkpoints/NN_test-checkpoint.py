import os
import numpy as np
from matplotlib import pyplot as plt
#matplotlib graphs will be included in the notebook, next to the code:
import random
#add PyTorch and TorchVision (used for cropping etc.)
import torch
import torchvision
import re
import json
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import pandas as pd
import seaborn as sns
from skorch import NeuralNet


def test_NN(run_no):
    run='run'+run_no
    data = json.load( open( "results/"+run+"/prop_"+run+".json" ) )
    # Hyperparameters
    BATCH_SIZE = data['batch_size']
    MOMENTUM = data['momentum']
    #k_folds = 4
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


    def test_model(model, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_losses, score = [], []
        total_score = 0
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.to(device))
                test_loss += F.mse_loss(output.to(device), target.to(device), size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                out=output.detach().cpu().numpy()
                targets=target.detach().cpu().numpy()
                r2 = pearsonr(targets.flatten(), out.flatten())[0]**2
                score.append(r2)
        total_score = np.mean(score)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        logging.info('\nTest set: Avg. loss: {:.4f}, R2 score: {:.4f}\n'.format(test_loss, total_score))

        return test_losses, total_score

    def reset_weights(m):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                #print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())


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

    #run1214 = np.hstack([run_12, run_14])
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
        
    df = pd.read_pickle('~/Documents/GitHub/EuXFEL-Virtual-Diagnostics/data/merged.pkl')
    df = df.filter(items=run_, axis=0)
    df = df.drop('date', axis=1)
    df = df.iloc[::n, :]

    features = data['features']
    targets = data['targets']
    inputs_outputs = features+targets
    dfmin = pd.Series(index=inputs_outputs, data=data['norm_min'])
    dfmax = pd.Series(index=inputs_outputs, data=data['norm_max'])
    testdf = df[inputs_outputs]
    normtest_df=(testdf-dfmin)/(dfmax-dfmin)

    HIDDEN_NODES = data['hidden_nodes']
    HIDDEN_LAYERS =data['hidden_layers']
    LEARNING_RATE = data['learning_rate']
    EPOCHS = data['epochs']



    #MOMENTUM = gs.best_params_['optimizer__momentum'] # 0.4
    logging.info('Testing NN:  Hidden layers: %s Hidden nodes: %s' % (HIDDEN_LAYERS, HIDDEN_NODES))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(HIDDEN_NODES, HIDDEN_LAYERS)

    model.load_state_dict(torch.load(cwd+f'/results/{run}/model-meas-fold-{run}.pth'))
    model.eval()
    #testdataset = torch.utils.data.TensorDataset(torch.tensor(normtest_df[features].values.astype(np.float32)), torch.tensor(normtest_df[targets].values.astype(np.float32)))
    #val_loader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE)



    outp = model(torch.tensor(normtest_df[features].values.astype(np.float32))).detach().numpy()

    val = {}
    r2 = {}
    rmse_val = {}
    for idx, target in enumerate(targets):
        val[target] = (outp[:,idx]*(dfmax[target]-dfmin[target])+dfmin[target])
    valpredict = pd.DataFrame(val, columns=targets)
    inputs = torch.tensor(testdf[features].values.astype(np.float32))


    for target in targets:
        r2[target] = pearsonr(np.asarray(testdf[target]), np.asarray(valpredict[target]))[0]**2
        rmse_val[target] = rmse(np.asarray(valpredict[target]), np.asarray(testdf[target]))

    logging.info('Evaluation with testing dataset: %s' % r2)