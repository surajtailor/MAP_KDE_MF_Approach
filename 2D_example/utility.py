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
        # calculate the covariance matrix
        self.data = data
        self.H = H
        self.beta = beta
        assert self.is_psd(self.H), "The covariance matrix H is not positive semi-definite."

    def is_psd(self, mat):
        return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real >= 0).all())

    def K(self, x):
        # calculate the scaling coeff to shorten the equation
        coeff = torch.linalg.det(self.H) ** (-0.5)
        # calculate the exponential term
        exp = torch.exp(-0.5 * torch.sum((x @ torch.inverse(self.H)) * x, dim=1))
        # pi term
        pi = torch.tensor(2 * torch.pi) ** (-x.shape[1] / 2)
        # return the kernel value
        return coeff * exp * pi

    def KDE(self, x):
        # prepare the grid for output values
        output = torch.zeros_like(x[:, 0])
        # process each sample
        for sample in self.data:
            output += self.K(x - sample.reshape(1, -1))
        # return the average
        output /= len(self.data)
        return output

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

def split_into_domain_bins_1D(x_data, z_data, x_data_scaled, z_data_scaled, n_bins, domain, device="cpu"):
    x_data = x_data.to(device)
    z_data = z_data.to(device)
    x_data_scaled = x_data_scaled.to(device)
    z_data_scaled = z_data_scaled.to(device)

    # Define bin edges along x₁
    bin_edges = torch.linspace(domain[0], domain[1], n_bins + 1, device=device)

    # Digitize x₁ values into bins
    bin_indices = torch.bucketize(x_data[:, 1], bin_edges, right=False) - 1  # -1 to make bins 0-based

    # Clamp any points exactly at domain[1] to last bin
    bin_indices = torch.clamp(bin_indices, min=0, max=n_bins - 1)

    bins = []
    bins_scaled = []
    for i in range(n_bins):
        idx = (bin_indices == i)
        bins.append((x_data[idx], z_data[idx]))
        bins_scaled.append((x_data_scaled[idx], z_data_scaled[idx]))
    return bins, bins_scaled

def split_into_domain_bins_2D(x_data, z_data, x_data_scaled, z_data_scaled, n_bins, domain, device="cpu"):
    x_data = x_data.to(device)
    z_data = z_data.to(device)
    x_data_scaled = x_data_scaled.to(device)
    z_data_scaled = z_data_scaled.to(device)

    min_x1, max_x1, min_x2, max_x2 = domain

    # Define bin edges along x₁ and x₂
    x1_edges = torch.linspace(min_x1, max_x1, n_bins + 1, device=device)
    x2_edges = torch.linspace(min_x2, max_x2, n_bins + 1, device=device)

    # Digitize x₁ and x₂ values into bins
    x1_bin_indices = torch.bucketize(x_data[:, 0], x1_edges, right=False) - 1
    x2_bin_indices = torch.bucketize(x_data[:, 1], x2_edges, right=False) - 1

    # Clamp indices (handle boundary points exactly at max_x1, max_x2)
    x1_bin_indices = torch.clamp(x1_bin_indices, min=0, max=n_bins - 1)
    x2_bin_indices = torch.clamp(x2_bin_indices, min=0, max=n_bins - 1)

    bins = []
    bins_scaled = []

    for i in range(n_bins):
        for j in range(n_bins):
            idx = (x1_bin_indices == i) & (x2_bin_indices == j)
            bins.append((x_data[idx], z_data[idx]))
            bins_scaled.append((x_data_scaled[idx], z_data_scaled[idx]))

    return bins, bins_scaled

# Plot the KDE output 3d plot
def plot_kde_3d(kde, x_test_scaled):
    kde_output = 100*kde.transformed_output(x_test_scaled).squeeze()
    # Convert to numpy for plotting
    x = x_test_scaled[:, 0].numpy()
    y = x_test_scaled[:, 1].numpy()
    z = kde_output.numpy()
    # Create a grid for surface plot
    from scipy.interpolate import griddata
    import numpy as np
    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100),np.linspace(y.min(), y.max(), 100))
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    # Plot surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.title("$KDE(x)$")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        # rotate the plot
    ax.view_init(elev=30, azim=30)
    plt.show()