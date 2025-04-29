import torch
import numpy as np
from data_generation import generate_lofi_data, generate_hifi_data, generate_test_data, create_scalers, scale_data
from net import NetAnyFunctional2D
from training_functions import trainer
import os
from utility import GaussianKDE, beta_calculator, split_into_domain_bins_2D
import random

## Set device and seeds for numpy, torch and random
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cpu'
scale_data_switch = True

## Set directory name
direc_name = "experiment_2D"

## Generate Lo-Fi and Hi-Fi data
hifi_regions = [[(0.0, 1.0), (0.0, 1.0)]]
hifi_regions_num_points = [1600]

lofi_regions = [(0, 1)]
lofi_regions_num_points = [1600]

x_lofi_data, y_lofi_data = generate_lofi_data(lofi_regions, lofi_regions_num_points)
x_hifi_data, y_hifi_data = generate_hifi_data(hifi_regions, hifi_regions_num_points)
x_test_data, y_test_data = generate_test_data(test_num=1600)
x_valid_data, y_valid_data = generate_test_data(test_num=100)

# Create scalers
scalers = create_scalers(x_lofi_data, y_lofi_data, x_hifi_data, y_hifi_data)

# Scale data
x_hifi_scaled, y_hifi_scaled = scale_data(x_hifi_data, y_hifi_data, scalers)
x_lofi_scaled, y_lofi_scaled = scale_data(x_lofi_data, y_lofi_data, scalers)
x_test_scaled, y_test_scaled = scale_data(x_test_data, y_test_data, scalers)
x_valid_scaled, y_valid_scaled = scale_data(x_valid_data, y_valid_data, scalers)

# Concatenating the data
hifi_train_data = (x_hifi_scaled, y_hifi_scaled)
lofi_train_data = (x_lofi_scaled, y_lofi_scaled)
valid_data = (x_valid_scaled, y_valid_scaled)
test_data = (x_test_scaled, y_test_scaled)
x_Global_Scaler, y_Global_Scaler = scalers

# Place data into sections
bins, bins_scaled = split_into_domain_bins_2D(x_hifi_data, y_hifi_data, x_hifi_scaled, y_hifi_scaled, 3, [0, 1, 0, 1], device="cpu")

# Create all training sets
x_train_1 = torch.concat([bins_scaled[0][0], bins_scaled[1][0], bins_scaled[2][0]], dim=0)
x_train_2 = torch.concat([x_train_1, bins_scaled[6][0]], dim=0)
x_train_3 = torch.concat([x_train_2, bins_scaled[7][0]], dim=0)
x_train_4 = torch.concat([x_train_3, bins_scaled[8][0]], dim=0)
x_train_5 = torch.concat([x_train_4, bins_scaled[3][0]], dim=0)
x_train_6 = torch.concat([x_train_5, bins_scaled[5][0]], dim=0)
x_train_7 = torch.concat([x_train_6, bins_scaled[4][0]], dim=0)
x_train_sets = [x_train_1, x_train_2, x_train_3, x_train_4, x_train_5, x_train_6, x_train_7]

y_train_1 = torch.concat([bins_scaled[0][1], bins_scaled[1][1], bins_scaled[2][1]], dim=0)
y_train_2 = torch.concat([y_train_1, bins_scaled[6][1]], dim=0)
y_train_3 = torch.concat([y_train_2, bins_scaled[7][1]], dim=0)
y_train_4 = torch.concat([y_train_3, bins_scaled[8][1]], dim=0)
y_train_5 = torch.concat([y_train_4, bins_scaled[3][1]], dim=0)
y_train_6 = torch.concat([y_train_5, bins_scaled[5][1]], dim=0)
y_train_7 = torch.concat([y_train_6, bins_scaled[4][1]], dim=0)
y_train_sets = [y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7]

train_sets = [(x_train_1, y_train_1), (x_train_2, y_train_2), (x_train_3, y_train_3), (x_train_4, y_train_4),
              (x_train_5, y_train_5), (x_train_6, y_train_6), (x_train_7, y_train_7)]

# Create validation sets from which we determine parameters of KDE for
x_valid_1 = bins_scaled[0][0]
x_valid_2 = torch.concat([bins_scaled[0][0], bins_scaled[6][0]], dim=0)
x_valid_sets = [x_valid_1, x_valid_2]

y_valid_1 = bins_scaled[0][1]
y_valid_2 = torch.concat([bins_scaled[0][1], bins_scaled[6][1]], dim=0)
y_valid_sets = [y_valid_1, y_valid_2]

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
# Initially set interpolate_H to False, as we will set it to True later
model = NetAnyFunctional2D(input_size=2, num_layers=num_layers, hidden_size=hidden_size, device=device,
                           functional=functional, poly_degree=None, fourier_degree=None, poly_l_degree=None,
                           custom=True, activation_func=activation_func, interpolate_H=True).to(device)

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
H = torch.tensor([[0.005, 0], [0, 0.005]])
beta = torch.tensor(5.0)

for i in range(3):
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
            tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='D')
    torch.save(model, directory_2 + "/D_pretrain")

    run_id = trainer(model, train_sets[i], valid_data, experiment_name, "model", run_name=run_name, n_iter= D_sto_n_iter,
            learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
            tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='D_sto')
    torch.save(model, directory_2 + "/D_sto")

    # Load D_sto model
    run_id = trainer(model, test_data, valid_data, experiment_name, "model", hyperparameters = hyperparameters,  run_name=run_name, n_iter= H_sto_n_iter,
            learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
            tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='H_sto_single_prior')
    torch.save(model, directory_2 + "/H_sto_single_prior")

# # Second KDE hyper tuning
H = torch.tensor([[0.02, 0], [0, 0.005]])
beta = torch.tensor(5.0)

for i in range(3, 7):
    directory_2 = "{:s}/{:s}/train_set_{}".format(direc_name, functional, i).replace('.', '_')
    os.makedirs(directory_2, exist_ok=True)
    # Fit interpolating function
    model._create_interpolate_H(train_sets[i])

    kde = GaussianKDE(x_train_sets[i], H, beta=beta)
    kde_output = 100 * kde.transformed_output(x_test_scaled)

    # Hyperparameters
    n_H = kde_output.squeeze()  # Choosen to be sufficiently large/ what is that number
    sig_mismatch = 1
    sig_prior = 1

    # Pack into a list
    hyperparameters = [n_H, sig_prior, sig_mismatch, alpha_0, beta_0]

    run_id = trainer(model, train_sets[i], valid_data, experiment_name, "model", run_name=run_name, n_iter= D_n_iter,
            learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
            tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='D')
    torch.save(model, directory_2 + "/D_pretrain")

    run_id = trainer(model, train_sets[i], valid_data, experiment_name, "model", run_name=run_name, n_iter= D_sto_n_iter,
            learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
            tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='D_sto')
    torch.save(model, directory_2 + "/D_sto")

    # Load D_sto model
    run_id = trainer(model, test_data, valid_data, experiment_name, "model", hyperparameters = hyperparameters , run_name=run_name,n_iter= H_sto_n_iter,
            learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100,tau=0.5,
            tau_adjustment=False,batch_size=batch_size,num_workers=num_workers ,case='H_sto_single_prior')
    torch.save(model,directory_2+"/H_sto_single_prior")