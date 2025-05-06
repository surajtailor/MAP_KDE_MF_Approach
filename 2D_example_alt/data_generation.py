import torch
import matplotlib.pyplot as plt
from utility import Scaler
import numpy as np

# Define functions for generating data
def function(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return torch.tensor((x1 - 1) ** 2 + (2 * x2 ** 2 - x1) ** 2)

def lofi_function(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return torch.tensor(0.8 * ((x1 - 1) ** 2 + (2 * x2 ** 2 - x1) ** 2) - 0.4 * x1 * x2 - 50)

def generate_lofi_data(lofi_regions, lofi_regions_num_points):
    lofi_total_num_points = sum(lofi_regions_num_points)
    lofi_regions_noise_stds = [0.01]
    lofi_regions_noise_mu = [0]

    x_lofi_data = torch.zeros((lofi_total_num_points, 2))
    y_lofi_data = torch.zeros((lofi_total_num_points,))

    start_idx = 0
    for region, num_points, mu, std in zip(lofi_regions, lofi_regions_num_points, lofi_regions_noise_mu, lofi_regions_noise_stds):
        grid_size = int(num_points ** 0.5)
        x1 = torch.linspace(region[0], region[1], grid_size)
        x2 = torch.linspace(region[0], region[1], grid_size)
        x1, x2 = torch.meshgrid(x1, x2)
        x = torch.stack((x1.flatten(), x2.flatten()), 1)
        y_lofi = torch.tensor(lofi_function(x)) + torch.normal(mean=mu, std=std, size=(x.size(0),))

        end_idx = start_idx + x.size(0)
        x_lofi_data[start_idx:end_idx, :] = x
        y_lofi_data[start_idx:end_idx] = y_lofi
        start_idx = end_idx

    return x_lofi_data, y_lofi_data

def generate_hifi_data(hifi_regions, hifi_regions_num_points):
    hifi_total_num_points = sum(hifi_regions_num_points)
    x_hifi_data = torch.zeros((hifi_total_num_points, 2))
    y_hifi_data = torch.zeros((hifi_total_num_points,))

    start_idx = 0
    for region, num_points in zip(hifi_regions, hifi_regions_num_points):
        x1 = region[0][0] + (region[0][1] - region[0][0]) * torch.rand(num_points)
        x2 = region[1][0] + (region[1][1] - region[1][0]) * torch.rand(num_points)
        x = torch.stack((x1, x2), 1)
        y_hifi = torch.tensor(function(x)) + torch.normal(mean=0, std=0.01, size=(num_points,))

        end_idx = start_idx + num_points
        x_hifi_data[start_idx:end_idx, :] = x
        y_hifi_data[start_idx:end_idx] = y_hifi
        start_idx = end_idx

    return x_hifi_data, y_hifi_data

def generate_test_data(test_num=1000):
    test_regions = (0, 1)
    # Test data
    test_grid_size = int(test_num ** 0.5)
    x1_test_data = torch.linspace(test_regions[0], test_regions[1], test_grid_size)
    x2_test_data = torch.linspace(test_regions[0], test_regions[1], test_grid_size)
    x1_test_data_grid, x2_test_data_grid = torch.meshgrid(x1_test_data, x2_test_data)
    x_test_data = torch.stack((x1_test_data_grid.flatten(), x2_test_data_grid.flatten()), 1)
    y_test_data = torch.tensor(function(x_test_data))

    return x_test_data, y_test_data, x1_test_data_grid, x2_test_data_grid

def create_scalers(x_lofi_data, y_lofi_data, x_hifi_data, y_hifi_data):
    # Scale data #Use a single scaler for all the data.
    x_Global_Scaler = Scaler()
    y_Global_Scaler = Scaler()

    # Fit to lofi data
    x_Global_Scaler.fit(torch.concatenate((x_lofi_data, x_hifi_data), 0))
    y_Global_Scaler.fit(torch.concatenate((y_lofi_data, y_hifi_data), 0))

    # Store scalers
    scalers = [x_Global_Scaler, y_Global_Scaler]
    return scalers

def create_scalers_single(x_lofi_data, y_lofi_data):
    # Scale data #Use a single scaler for all the data.
    x_Global_Scaler = Scaler()
    y_Global_Scaler = Scaler()

    # Fit to lofi data
    x_Global_Scaler.fit(x_lofi_data)
    y_Global_Scaler.fit(y_lofi_data)

    # Store scalers
    scalers = [x_Global_Scaler, y_Global_Scaler]
    return scalers

def scale_data(x_data, y_data, scalers):
    x_scaled = scalers[0].transform(x_data)
    y_scaled = scalers[1].transform(y_data)
    return x_scaled, y_scaled

def plot_data(x_data, y_data, title="Data Plot"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_data[:, 0], y_data)
    plt.title(title)
    plt.grid()
    plt.show()