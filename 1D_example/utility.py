import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from torch.utils.data import Dataset, DataLoader

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with data before transforming.")
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with data before inverse transforming.")
        return x * self.std + self.mean

    def inverse_transform_mu_D(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with data before inverse transforming.")
        return x * self.std

    def inverse_transform_variance(self, x):
        return x * self.std ** 2

class GaussianKDE:
    def __init__(self, data, H, beta=1):
        self.data = data.flatten()  # Ensure 1D data
        self.H = H  # Bandwidth as a scalar
        self.beta = beta
        assert self.H > 0, "Bandwidth H must be positive."

    def K(self, x):
        """ 1D Gaussian Kernel Function """
        coeff = (2 * torch.pi * self.H**2) ** -0.5  # Normalization term
        exp = torch.exp(-0.5 * (x / self.H) ** 2)  # Exponential term
        return coeff * exp

    def KDE(self, x):
        """ Kernel Density Estimation """
        output = torch.zeros_like(x)
        for sample in self.data:
            output += self.K(x - sample)  # Apply kernel to each sample
        return output / len(self.data)  # Normalize by number of data points

    def translated_sigmoid(self, x, gamma=1.0):
        """ Sigmoid Transformation """
        beta_scaled = torch.exp(gamma * self.beta)
        sigma_x = 1 / (1 + torch.exp(-beta_scaled * x))
        sigma_0 = 1 / (1 + torch.exp(torch.tensor(0.0)))
        return (sigma_x - sigma_0) / (1 - sigma_0)

    def transformed_output(self, x):
        """ Apply KDE and transformation """
        x = self.KDE(x)
        return self.translated_sigmoid(x)

    def __call__(self, x):
        return self.KDE(x)

def unconcatenate_mu(x):
    mu_H = x[:, 0]
    mu_L = x[:, 1]
    mu_D = x[:, 2]
    H = x[:, 3]
    return mu_H, mu_L, mu_D, H

def unconcatenate_var(x):
    var_H = x[:, 0]
    var_L = x[:, 1]
    var_D = x[:, 2]
    return var_H, var_L, var_D

def beta_calculator(alpha, mode):
    beta = (alpha + 1) * mode
    return beta

class IndexedDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx  # return index too