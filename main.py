import random
import pickle as pl
import numpy as np
from emnist_infrastructure import emnist_data
from pretrain import pretrain_model
from run_expt import run_blum, run_fedavg


def build_mixtures(high_idxs, low_idxs, train_size = 2000):
    """Build mixtures as describe in the paper. """
    training_samples = []
    validations_indices = []
    names = []
    for alpha in np.linspace(0, 0.1, 2):
        names.append(alpha)
        samps = random.sample(low_idxs, int(alpha * train_size)) + random.sample(high_idxs,
                                                                                 int((1 - alpha) * train_size))
        training_samples.append(random.sample(samps, int(0.8 * train_size)))
        validations_indices.append([i for i in samps if i not in training_samples[-1]])

    for alpha in np.linspace(.9, 1, 2):
        names.append(alpha)
        samps = random.sample(low_idxs, int(alpha * train_size)) + random.sample(high_idxs,
                                                                                 int((1 - alpha) * train_size))
        training_samples.append(random.sample(samps, int(0.8 * train_size)))
        validations_indices.append([i for i in samps if i not in training_samples[-1]])

    return training_samples, validations_indices, names

if  __name__ == "__main__":
    """I've included the main results of the code. The full experiment is somewhat computationally expensive as-is,
    and so I've left out most of the analyse in this branch for simplicity."""
    trained_model = pretrain_model(emnist_data)
    with open('high_and_low_indices', 'rb') as f:
        high_idxs, low_idxs = pl.load(f)
    training_samples, validation_indices, names = build_mixtures(high_idxs, low_idxs)
    blum_results = run_blum(trained_model, names, training_samples, validation_indices, emnist_data, trials=2)
    fedavg_results = run_fedavg(trained_model, names, training_samples, validation_indices, emnist_data, trials=2)
