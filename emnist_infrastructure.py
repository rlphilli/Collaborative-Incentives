from copy import deepcopy
from random import sample
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
        ])

emnist_data = datasets.EMNIST('data/', split='balanced', train=True, 
                              download=True, transform=preprocess)

