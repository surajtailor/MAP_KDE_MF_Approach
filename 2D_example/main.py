import torch
import numpy as np
from data_generation import generate_lofi_data, generate_hifi_data, generate_test_data, create_scalers, scale_data
from net import NetAnyFunctional2D
from training_functions import trainer
import os
from utility import GaussianKDE, beta_calculator
import random

## Set device and seeds for numpy, torch and random
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cpu'
scale_data_switch = True

## Set directory name
direc_name = "2d_example"

## Set plotting parameters
title_fontsize = 9
axis_fontsize = 9
legend_fontsize = 6
linewidth = 0.7
scatter_size = 2

## Generate Lo-Fi and Hi-Fi data
hifi_regions = [[(0.0, 0.25), (0.0, 0.25)], [(0.25, 0.5), (0.0, 0.25)], [(0.5, 0.75), (0.0, 0.25)], [(0.75, 1.0), (0.0, 0.25)], [(0.0, 0.25), (0.25, 0.5)], [(0.25, 0.5), (0.25, 0.5)], [(0.5, 0.75), (0.25, 0.5)], [(0.75, 1.0), (0.25, 0.5)], [(0.0, 0.25), (0.5, 0.75)], [(0.25, 0.5), (0.5, 0.75)], [(0.5, 0.75), (0.5, 0.75)], [(0.75, 1.0), (0.5, 0.75)], [(0.0, 0.25), (0.75, 1.0)], [(0.25, 0.5), (0.75, 1.0)], [(0.5, 0.75), (0.75, 1.0)], [(0.75, 1.0), (0.75, 1.0)]]
hifi_regions_num_points = [6, 6, 22, 6, 2, 1, 6, 4, 2, 1, 18, 4, 4, 2, 15, 1]

lofi_regions = [(0, 1)]
lofi_regions_num_points = [1000]

# Generate each subset of data
x_hifi_1, y_hifi_1 = generate_hifi_data(hifi_regions, hifi_regions_num_points)
x_hifi_2, y_hifi_2 = generate_hifi_data(hifi_regions, hifi_regions_num_points)
x_hifi_3, y_hifi_3 = generate_hifi_data(hifi_regions, hifi_regions_num_points)
x_hifi_4, y_hifi_4 = generate_hifi_data(hifi_regions, hifi_regions_num_points)

# Concatenate all the hifi data
x_hifi_data = torch.cat([x_hifi_1, x_hifi_2, x_hifi_3, x_hifi_4], dim=0)
y_hifi_data = torch.cat([y_hifi_1, y_hifi_2, y_hifi_3, y_hifi_4], dim=0)
x_lofi_data, y_lofi_data = generate_lofi_data(lofi_regions, lofi_regions_num_points)
x_test_data, y_test_data,_, _ = generate_test_data(test_num=1000)
x_valid_data, y_valid_data,_,_ = generate_test_data(test_num=100)

# Create scalers
scalers = create_scalers(x_lofi_data, y_lofi_data, x_hifi_1, y_hifi_1)

# Scale data
x_hifi_scaled, y_hifi_scaled = scale_data(x_hifi_data, y_hifi_data, scalers)
x_lofi_scaled, y_lofi_scaled = scale_data(x_lofi_data, y_lofi_data, scalers)
x_test_scaled, y_test_scaled = scale_data(x_test_data, y_test_data, scalers)
x_valid_scaled, y_valid_scaled = scale_data(x_valid_data, y_valid_data, scalers)
x_hifi_scaled_1, y_hifi_scaled_1 = scale_data(x_hifi_1, y_hifi_1, scalers)
x_hifi_scaled_2, y_hifi_scaled_2 = scale_data(x_hifi_2, y_hifi_2, scalers)
x_hifi_scaled_3, y_hifi_scaled_3 = scale_data(x_hifi_3, y_hifi_3, scalers)
x_hifi_scaled_4, y_hifi_scaled_4 = scale_data(x_hifi_4, y_hifi_4, scalers)

# Concatenating the data
hifi_train_data = (x_hifi_scaled, y_hifi_scaled)
lofi_train_data = (x_lofi_scaled, y_lofi_scaled)
valid_data = (x_valid_scaled, y_valid_scaled)
test_data = (x_test_scaled, y_test_scaled)
x_Global_Scaler, y_Global_Scaler = scalers

# Create all training sets
x_train_1 = x_hifi_scaled_1
x_train_2 = torch.concat([x_hifi_scaled_1, x_hifi_scaled_2], dim=0)
x_train_3 = torch.concat([x_train_2, x_hifi_scaled_3], dim=0)
x_train_4 = torch.concat([x_train_3, x_hifi_scaled_4], dim=0)
x_train_sets = [x_train_1, x_train_2, x_train_3, x_train_4]

y_train_1 = y_hifi_scaled_1
y_train_2 = torch.concat([y_train_1, y_hifi_scaled_2], dim=0)
y_train_3 = torch.concat([y_train_2, y_hifi_scaled_3], dim=0)
y_train_4 = torch.concat([y_train_3, y_hifi_scaled_4], dim=0)
y_train_sets = [y_train_1, y_train_2, y_train_3, y_train_4]

# Place into tuples
train_sets = [(x_train_1, y_train_1), (x_train_2, y_train_2), (x_train_3, y_train_3), (x_train_4, y_train_4)]

## Non Sto iterations
H_n_iter = 1000
L_n_iter = 1000
D_n_iter = 1000

## Sto Iterations
L_sto_n_iter = 2000
D_sto_n_iter = 2000
H_sto_n_iter = 10000

lr = 1e-3
tau = 1
num_workers = 0
batch_size = 40

# Num layers and hidden units
num_layers = 12
hidden_size = 40
activation_func = 'relu'

# Fixed hyperparameters
mode_var_h = 0.5  # Value of uncertainty that it defaults to, after calculating the prior.
alpha_0 = 1  # Can set this to whatever we want.
beta_0 = beta_calculator(alpha_0, mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.
if scale_data_switch == True:
    mode_var_h = mode_var_h / (y_Global_Scaler.std ** 2)
    beta_0 = beta_calculator(alpha_0, mode_var_h)

# Custom Functional
functional = 'custom_1'
model = NetAnyFunctional2D(input_size=2, num_layers=num_layers, hidden_size=hidden_size, device=device,
                           functional=functional, poly_degree=None, fourier_degree=None, poly_l_degree=None,
                           custom=True, activation_func=activation_func, interpolate_H=True).to(device) # MODEL HAS BEEN INITIALISED ALREADY IN THE CLASS. MAKE SURE TO CHANGE

# Create results directory
directory = "{:s}/{:s}".format(direc_name, functional).replace('.', '_')
os.makedirs(directory, exist_ok=True)
run_name = "1"
experiment_name = directory

# Initialise the H model on all training data
model._create_interpolate_H(hifi_train_data)

run_id = trainer(model, lofi_train_data, valid_data, experiment_name, "model", run_name=run_name, n_iter=L_n_iter,
                 learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
                 tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='L')
torch.save(model, directory + "/L_pretrain")

# Stochastic Training
run_id = trainer(model, lofi_train_data, valid_data, experiment_name, "model", run_name=run_name, n_iter=L_sto_n_iter,
                 learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
                 tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='L_sto')
torch.save(model, directory + "/L_sto")

# First KDE choice
H = torch.tensor([[0.003, 0], [0, 0.003]])
beta = torch.tensor(3.0)

for i in range(4):
    directory_2 = "{:s}/{:s}/train_set_{}".format(direc_name, functional, i).replace('.', '_')
    os.makedirs(directory_2, exist_ok=True)

    # Fit interpolating function
    model._create_interpolate_H(train_sets[i])

    # Fit KDE
    kde = GaussianKDE(x_train_sets[i], H, beta=beta)
    kde_output = kde.transformed_output(x_test_scaled)

    # Hyperparameters
    n_H = 100 * kde_output.squeeze()  # Choosen to be sufficiently large/ what is that number
    sig_mismatch = 1
    sig_prior = 1

    # Pack into a list
    hyperparameters = [n_H, sig_prior, sig_mismatch, alpha_0, beta_0]

    run_id = trainer(model, train_sets[i], valid_data, experiment_name, "model", run_name=run_name, n_iter= D_n_iter,
            learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
            tau_adjustment=False, batch_size=1, num_workers=num_workers, case='D')
    torch.save(model, directory_2 + "/D_pretrain")

    run_id = trainer(model, train_sets[i], valid_data, experiment_name, "model", run_name=run_name, n_iter= D_sto_n_iter,
            learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
            tau_adjustment=False, batch_size=1, num_workers=num_workers, case='D_sto')
    torch.save(model, directory_2 + "/D_sto")

    # Load D_sto model
    run_id = trainer(model, test_data, valid_data, experiment_name, "model", hyperparameters = hyperparameters,  run_name=run_name, n_iter= H_sto_n_iter,
            learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
            tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='H_sto_single_prior')
    torch.save(model, directory_2 + "/H_sto_single_prior")