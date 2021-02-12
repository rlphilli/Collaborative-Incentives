import numpy as np
import pickle as pl
import torch.nn as nn
import torch.optim as optim

from learners import Net, FEDERATOR_Avg, label_array, training_array


def pretrain_model(emnist_data
):
  """Pre-train a net object."""
  FB = FEDERATOR_Avg(Net(), training_array(emnist_data), [list(range(len(emnist_data) - 30000, len(emnist_data)))],
                     label_array(emnist_data),
                     model_names=['5'], validation_indices=[list(range(60000, len(emnist_data) - 30000))])
  FB.initialize_client_models(optim.Adam, nn.CrossEntropyLoss(), 0.002)
  print('initialized')
  FB.initialize_weights(np.ones(1))

  for i in range(40):
    FB.federated_train(64, 1, 1, eval_models=True,  epsilon=0.1)

  return FB.model
