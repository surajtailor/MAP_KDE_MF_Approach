import torch
import numpy as np
from data_generation import generate_lofi_data, generate_hifi_data, generate_test_data, create_scalers, scale_data
from net import NetAnyFunctional
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

## Set directory name
direc_name = "main"

## Generate Lo-Fi and Hi-Fi data
hifi_regions_hyper_train = [[(0, 0.1)]]
hifi_regions_num_points_hyper_train = [50]

hifi_regions_hyper_valid = [[(0.4, 0.5)]]
hifi_regions_num_points_hyper_valid = [50]

hifi_regions = [[(0,.25)], [(0.75,1)]]
hifi_regions_num_points = [500, 500]

lofi_regions = [(0, 1)]
lofi_regions_num_points = [1000]

x_lofi_data, y_lofi_data = generate_lofi_data(lofi_regions, lofi_regions_num_points)
x_hyper_data, y_hyper_data = generate_hifi_data(hifi_regions_hyper_train, hifi_regions_num_points_hyper_train)
x_hyper_valid_data, y_hyper_valid_data = generate_hifi_data(hifi_regions_hyper_valid, hifi_regions_num_points_hyper_valid)
x_hifi_data, y_hifi_data = generate_hifi_data(hifi_regions, hifi_regions_num_points)
x_test_data, y_test_data = generate_test_data(test_num=1000)
x_valid_data, y_valid_data = generate_test_data(test_num=100)

# Create scalers
scalers = create_scalers(x_lofi_data, y_lofi_data, x_hifi_data, y_hifi_data)

# Scale data
x_hifi_scaled, y_hifi_scaled = scale_data(x_hifi_data, y_hifi_data, scalers)
x_lofi_scaled, y_lofi_scaled = scale_data(x_lofi_data, y_lofi_data, scalers)
x_hyper_scaled, y_hyper_scaled = scale_data(x_hyper_data, y_hyper_data, scalers)
x_hyper_valid_scaled, y_hyper_valid_scaled = scale_data(x_hyper_valid_data, y_hyper_valid_data, scalers)
x_test_scaled, y_test_scaled = scale_data(x_test_data, y_test_data, scalers)
x_valid_scaled, y_valid_scaled = scale_data(x_valid_data, y_valid_data, scalers)

# Concatenating the data
lofi_train_data = (x_lofi_scaled, y_lofi_scaled)
valid_data = (x_valid_scaled, y_valid_scaled)
x_Global_Scaler, y_Global_Scaler = scalers

# Concatenating the data
hifi_train_data = (x_hifi_scaled, y_hifi_scaled)
lofi_train_data = (x_lofi_scaled, y_lofi_scaled)
valid_data = (x_valid_scaled, y_valid_scaled)
test_data = (x_test_scaled, y_test_scaled)
x_Global_Scaler, y_Global_Scaler = scalers

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

# Set List of Functionals
polynomial_degree_list_overlap = [1]
polynomial_on_l_degree_list_overlap = [3]


# #Poly and Poly on Mu L Combined Functionals
for poly_degree in polynomial_degree_list_overlap:
    for poly_on_l_degree in polynomial_on_l_degree_list_overlap:
        functional = 'polynomial_polynomial_on_l'
        model = NetAnyFunctional(input_size=1, num_layers=num_layers, hidden_size=hidden_size, device=device,
                                     functional=functional, poly_degree=poly_degree, fourier_degree=None, poly_l_degree=poly_on_l_degree,
                                     activation_func=activation_func, interpolate_H=True).to(device)

        # Create results directory
        directory = "{:s}/{:s}/degree_poly_{:s}_degree_poly_l_{:s}".format(direc_name, functional, str(poly_degree), str(poly_on_l_degree)).replace('.', '_')
        os.makedirs(directory, exist_ok=True)
        run_name = "1"
        experiment_name = directory

        # Fit interpolating function
        model._create_interpolate_H(hifi_train_data)
        torch.save(model, directory + "/H")

        run_id = trainer(model, lofi_train_data, valid_data, experiment_name, "model", run_name=run_name, n_iter= L_n_iter,
                learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
                tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='L')
        torch.save(model, directory + "/L_pretrain")

        run_id = trainer(model, hifi_train_data, valid_data, experiment_name, "model", run_name=run_name, n_iter= D_n_iter,
                learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
                tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='D')
        torch.save(model, directory + "/D_pretrain")

        # Stochastic Training
        run_id = trainer(model, lofi_train_data, valid_data, experiment_name, "model", run_name=run_name, n_iter= L_sto_n_iter,
                learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
                tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='L_sto')
        torch.save(model, directory + "/L_sto")

        run_id = trainer(model, hifi_train_data, valid_data, experiment_name, "model", run_name=run_name, n_iter= D_sto_n_iter,
                learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
                tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='D_sto')
        torch.save(model, directory + "/D_sto")

# Default parameters
H_band = torch.tensor(0.01)
beta = torch.tensor(3)

# Initialise KDE
kde = GaussianKDE(x_hifi_scaled, H_band, beta=beta)
kde_output = kde.transformed_output(x_test_scaled)

# Hyperparameters
n_H = 100*kde_output.squeeze() # Choosen to be sufficiently large/ what is that number
sig_mismatch = 1
sig_prior = 1 # Can ignore

alpha_0 = 1  # Can set this to whatever we want.
mode_var_h = 1  # Value of uncertainty that it defaults to, after calculating the prior.
beta_0 = beta_calculator(alpha_0, mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.

if scale_data == True:
    mode_var_h = mode_var_h / (y_Global_Scaler.std ** 2)
    beta_0 = beta_calculator(alpha_0, mode_var_h)

# Pack into a list
hyperparameters = [n_H, sig_prior, sig_mismatch, alpha_0, beta_0]

#Varying the bandwidth parameter. Fixing beta = 3
model = torch.load(directory + "/D_sto")
for k in [0.01, 0.1, 0.15, 0.2]:
    # Default parameters
    H_band = torch.tensor(k)
    beta = torch.tensor(3)
    kde = GaussianKDE(x_hifi_scaled, H_band, beta=beta)
    kde_output = kde.transformed_output(x_test_scaled)

    # Hyperparameters
    n_H = 100 * kde_output.squeeze()  # Choosen to be sufficiently large/ what is that number
    sig_mismatch = 1
    sig_prior = 1  # Can ignore

    alpha_0 = 1  # Can set this to whatever we want.
    mode_var_h = 1  # Value of uncertainty that it defaults to, after calculating the prior.
    beta_0 = beta_calculator(alpha_0, mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.

    # Pack into a list
    hyperparameters = [n_H, sig_prior, sig_mismatch, alpha_0, beta_0]

    # Load D_sto model
    run_id = trainer(model, test_data, valid_data, experiment_name, "model", hyperparameters = hyperparameters,  run_name=run_name, n_iter= H_sto_n_iter,
            learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100, tau=0.5,
            tau_adjustment=False, batch_size=batch_size, num_workers=num_workers, case='H_sto_single_prior')
    torch.save(model, directory + "/final_H_band_{}".format(k))

# Varying kappa paramteter
model = torch.load(directory + "/D_sto")
for k in [0.1, 1.0, 10.0, 100.0]:
    # Default parameters
    H_band = torch.tensor(0.01)
    beta = torch.tensor(3)
    kde = GaussianKDE(x_hifi_scaled, H_band, beta=beta)
    kde_output = kde.transformed_output(x_test_scaled)

    # Hyperparameters
    n_H = k * kde_output.squeeze()  # Choosen to be sufficiently large/ what is that number
    sig_mismatch = 1
    sig_prior = 1  # Can ignore

    alpha_0 = 1  # Can set this to whatever we want.
    mode_var_h = 1  # Value of uncertainty that it defaults to, after calculating the prior.
    beta_0 = beta_calculator(alpha_0,mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.

    # Pack into a list
    hyperparameters = [n_H, sig_prior, sig_mismatch, alpha_0, beta_0]

    # Load D_sto model
    run_id = trainer(model, test_data, valid_data, experiment_name, "model", hyperparameters=hyperparameters,
                     run_name=run_name, n_iter=H_sto_n_iter,
                     learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100,
                     tau=0.5,
                     tau_adjustment=False, batch_size=batch_size, num_workers=num_workers,
                     case='H_sto_single_prior')
    torch.save(model, directory + "/final_kappa_{}".format(k))

# Varying beta_0 prior paramteter
model = torch.load(directory + "/D_sto")
for k in [0.1, 0.4, 0.7, 1.0]:
    # Default parameters
    H_band = torch.tensor(0.01)
    beta = torch.tensor(3)
    kde = GaussianKDE(x_hifi_scaled, H_band, beta=beta)
    kde_output = kde.transformed_output(x_test_scaled)

    # Hyperparameters
    n_H = 100 * kde_output.squeeze()  # Choosen to be sufficiently large/ what is that number
    sig_mismatch = 1
    sig_prior = 1  # Can ignore

    alpha_0 = 1  # Can set this to whatever we want.
    mode_var_h = k  # Value of uncertainty that it defaults to, after calculating the prior.
    beta_0 = beta_calculator(alpha_0,mode_var_h)  # This is the value of beta_0 that will give us the expected value of var_H.

    # Pack into a list
    hyperparameters = [n_H, sig_prior, sig_mismatch, alpha_0, beta_0]

    # Load D_sto model
    run_id = trainer(model, test_data, valid_data, experiment_name, "model", hyperparameters=hyperparameters,
                     run_name=run_name, n_iter=H_sto_n_iter,
                     learning_rate=lr, scheduler_type=None, save_freq=1000, weight_decay=0, valid_freq=100,
                     tau=0.5,
                     tau_adjustment=False, batch_size=batch_size, num_workers=num_workers,
                     case='H_sto_single_prior')
    torch.save(model, directory + "/final_beta_0_{}".format(beta_0))