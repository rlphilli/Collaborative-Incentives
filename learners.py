from copy import deepcopy
import random
from random import sample, choices
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class training_array(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset
  def __getitem__(self, idx):
    if isinstance(idx, list):
        return np.array([self.dataset[i][0].numpy() for i in idx])
    else:
        return self.dataset[i][0]
  def __len__(self):
      return len(self.dataset)
class label_array:
  def __init__(self, dataset):
    self.dataset = dataset
  def __getitem__(self, idx):
    if isinstance(idx, list):
        return np.array([self.dataset[i][1] for i in idx])
    else:
        return self.dataset[idx][1]
  def __len__(self):
    return self.dataset.shape[0]

    
class Net(nn.Module):
    """Defines a pretty solid, albeit simple, CNN for the balanced EMNIST task"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.drop2 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(128, 47)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class client:
    def __init__(self, model_architecture, optimizer, loss, name):
        """Instantiate a client model with relevant training information"""
        self.model = model_architecture
        self.old_model = None
        self.opt = optimizer
        self.loss = loss
        self.contrib = []
        self.model.to('cuda')
        self.name = name

    def snapshot_model(self):
        self.old_model = deepcopy(self.model)
    
    def train(self, x, y, model_state_dict):
        """Train the client on a set of data and a given model dict. Only used in the 
        multi-batch federated algorithms"""
        self.model.load_state_dict(model_state_dict)
        if len(self.contrib) >= 1:
          self.contrib.append(y.shape[0] + self.contrib[-1])
        else:
          self.contrib.append(y.shape[0])
        
        self.model.train(True)
        self.model.to('cuda')
        self.opt.zero_grad()

        out = self.model(x.to(self.device))
        loss = self.loss(out, y)
        loss.backward()
        self.opt.step()
    
    def set_model(self, model_dict):
        self.model.load_state_dict(model_dict)
        
    def model_restore(self):
        self.model = self.old_model

class Federated_Server:
    """
    Args:
        model (torch.nn) : pytorch model
        dataset (torch.tensor) : 
        y_array (torch.tensor) : 
        dataset_indices (list of list) :
        model_names (list of str) :

    """
    def __init__(self, model, dataset, dataset_indices, y_array, model_names = None, validation_indices = None):
        """Instantiates a federated_server"""
        self.model = model
        self.dataset = dataset
        self.model_names = model_names
        if self.model_names:
            if len(dataset_indices) != len(model_names):
                raise Exception('Provided list of dataset indices should match the length of model names')
        else:
            self.model_names = range(len(dataset_indices))
            
        self.model_indices = dict(zip(self.model_names, dataset_indices))
        self.y_array = y_array
        
        if validation_indices:
            self.validation_indices = validation_indices
            
  
    def initialize_client_models(self, optimizer, loss, learning_rate):
        """Initialize client models"""
        sd = self.model.state_dict()
        self.learning_rate = learning_rate
        self.models = {}
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.loss = loss

        for i in self.model_names:
            print('initializing', i)
            local_model = deepcopy(self.model)
            local_optimizer = optimizer(local_model.parameters(), lr=learning_rate)
            local_client = client(local_model, local_optimizer, loss, name=i)
            self.models[i] = {'client': local_client}
            self.models[i]['training points'] = self.model_indices[i]
            self.models[i]['validation points'] = self.model_indices[i]
            self.models[i]['training loss'] = []
            self.models[i]['validation loss'] = []
            self.models[i]['validation accuracy'] = []

    def federated_train(self):
        raise NotImplementedError
        
    def __repr__(self):
        return "Federated server of " + str(self.model_names)
    
    def evaluate_on_validation(self):
        """Evaluate each model on a list of indices that correspond to that model's validation set"""
        for model_name, validation_idxs in zip(self.model_names, self.validation_indices):
            validation_idxs = sample(validation_idxs, min(200, len(validation_idxs)))
            dataset_to_eval = self.dataset[validation_idxs] 
            y_to_eval = torch.from_numpy(self.y_array[validation_idxs])
            dataset_to_eval = torch.from_numpy(dataset_to_eval).float()
            model_to_eval = self.model
            model_to_eval.eval()
            outs = []
            self.optimizer.zero_grad()
            with torch.no_grad():
              out = model_to_eval(dataset_to_eval.to('cuda'))

            loss = self.loss(out.to('cpu'), y_to_eval.to('cpu'))
            accuracy = torch.argmax(F.softmax(out, dim=1),1)
            
            accuracy = torch.eq(accuracy.to('cpu'), y_to_eval)

            accuracy = torch.sum(accuracy) / len(validation_idxs)

            self.models[model_name]['validation loss'].append(loss)
            self.models[model_name]['validation accuracy'].append(accuracy) 


class Blum_Avg(Federated_Server):
    """      
    Args:
        model (torch.nn) : pytorch model
        dataset (torch.tensor)  
        y_array (torch.tensor) 
        dataset_indices (list of list) 
        model_names (list of str) 

    """
    def __init__(self, model, dataset, dataset_indices, y_array, validation_indices, model_names = None):
        super().__init__(model, dataset, dataset_indices, y_array, model_names, validation_indices)

    def initialize_weights(self):
      self.weights = np.ones(len(self.models))/len(self.models)
    def federated_train(self, batch_size, epochs, local_batches, eval_models = False, epsilon=0.3 ):
        for eeee in range(epochs):

            if len(self.models[self.model_names[0]]['validation accuracy']) == 0:
              test_results = [0 for _ in self.models]
            else:
              test_results = [self.models[i]['validation accuracy'][-1].item() for i in self.models]
            tt1 = deepcopy(test_results)

            test_results = [4 if i<=(1-epsilon) else 1 for i in test_results]
            self.tr = set(test_results) == {1}

            self.weights = self.weights*test_results/np.linalg.norm(self.weights*test_results, 1)
            samples_to_take = self.weights*batch_size*local_batches*len(self.models) 
            samples_to_take = np.round(samples_to_take).tolist()
            assert(local_batches == 1)
            assert( np.abs(1- np.sum(self.weights) <= 0.1))

            model_names = []
            contribs_this_round = []
            train_indices = []
            for model, sample_alloc in zip(self.model_names, samples_to_take):
                model_names.append(model)
                contribs_this_round.append(0)
                a_model = self.models[model]
                train_indices += choices(a_model['training points'], k=int(sample_alloc))

                a_model['client'].contrib.append(sample_alloc)

            y = torch.from_numpy(self.y_array[train_indices])
            x = self.dataset[train_indices]
            x = torch.from_numpy(x).float()

            self.optimizer.zero_grad()
            self.model.train()
            out = self.model(x.to('cuda'))
            loss = self.loss(out, y.to('cuda'))
            loss.backward()
            self.optimizer.step()

            self.evaluate_on_validation()


class FEDERATOR_Avg(Federated_Server):
    """      
    Args:
        model (torch.nn) : pytorch model
        dataset (torch.tensor)  
        y_array (torch.tensor) 
        dataset_indices (list of list) 
        model_names (list of str) 

    """
    def __init__(self, model, dataset, dataset_indices, y_array, validation_indices, model_names = None):
        print('hereitis', len(dataset), len(dataset_indices), len(validation_indices))
        print(len(model_names))
        super().__init__(model, dataset, dataset_indices, y_array, model_names, validation_indices)

    def initialize_weights(self, weights=None ):
      self.weights = weights

    def federated_train(self, batch_size, epochs, local_batches, eval_models = False, epsilon=0.3):
        """"""
        # 

        for eeee in range(epochs):
            # self.evaluate_on_validation()
            # Get last model results from above evaluation
            if len(self.models[self.model_names[0]]['validation accuracy']) == 0:
              test_results = [0 for _ in self.models]
            else:
              test_results = [self.models[i]['validation accuracy'][-1].item() for i in self.models]

            self.tr = set(test_results) == {1}

            samples_to_take = self.weights*batch_size*local_batches*len(self.models) 
            samples_to_take = np.round(samples_to_take).tolist()
            assert(local_batches == 1)
            assert( np.abs(1- np.sum(self.weights) <= 0.1))

            model_names = []
            contribs_this_round = []
            train_indices = []
            for model, sample_alloc in zip(self.model_names, samples_to_take):
                model_names.append(model)
                contribs_this_round.append(0)
                a_model = self.models[model]
                train_indices += random.choices(a_model['training points'], k=int(sample_alloc))
                a_model['client'].contrib.append(sample_alloc)

            y = torch.from_numpy(self.y_array[train_indices])
            x = self.dataset[train_indices]
            x = torch.from_numpy(x).float()

            self.optimizer.zero_grad()
            self.model.train()
            out = self.model(x.to('cuda'))
            loss = self.loss(out, y.to('cuda'))
            loss.backward()

            self.optimizer.step()

            self.evaluate_on_validation()



              
