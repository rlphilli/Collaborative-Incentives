from copy import deepcopy
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from learners import FEDERATOR_Avg, Blum_Avg, training_array, label_array


def run_blum(trained_model, names, training_samples, validation_indices, emnist_data, trials=100, batch_size=256):
    for i in range(trials):
        for agent in range(len(names)):
            results = []
            for perc in [1, 0.9, 0.75, 0.5, 0.25, 0.05]:
                local_train = deepcopy(training_samples)
                local_train[agent] = random.sample(local_train[agent], int(len(local_train[agent]) * perc))

                FA = Blum_Avg(deepcopy(trained_model), training_array(emnist_data), local_train, label_array(emnist_data),
                              model_names=names, validation_indices=validation_indices)
                FA.initialize_client_models(optim.Adam, nn.CrossEntropyLoss(), 0.002)

                FA.initialize_weights()
                idx = 0
                while idx <= 75:
                    idx += 1
                    FA.federated_train(batch_size, 1, 1, eval_models=True, epsilon=0.3)
                results.append([i, agent, perc, [FA.models[j]['validation accuracy'] for j in FA.model_names],
                                   [FA.models[j]['client'].contrib for j in FA.model_names]])
    return results

def run_fedavg(trained_model, names, training_samples, validation_indices, emnist_data, trials=100, batch_size=256):
    for i in range(trials):
        for agent in range(len(names)):
            results = []
            for perc in [1, 0.9, 0.75, 0.5, 0.25, 0.05]:
                local_train = deepcopy(training_samples)
                local_train[agent] = random.sample(local_train[agent], int(len(local_train[agent]) * perc))

                FA = FEDERATOR_Avg(deepcopy(trained_model), training_array(emnist_data), local_train, label_array(emnist_data),
                              model_names=names, validation_indices=validation_indices)
                FA.initialize_client_models(optim.Adam, nn.CrossEntropyLoss(), 0.002)
                weighting = np.ones(len(names))

                FA.initialize_weights(weighting/len(names))
                idx = 0
                while idx <= 75:
                    idx += 1
                    FA.federated_train(batch_size, 1, 1, eval_models=True, epsilon=0.0)
                results.append([i, agent, perc, [FA.models[j]['validation accuracy'] for j in FA.model_names],
                                   [FA.models[j]['client'].contrib for j in FA.model_names]])
    return results