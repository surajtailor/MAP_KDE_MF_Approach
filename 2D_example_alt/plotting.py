import torch
import numpy as np
from data_generation import generate_lofi_data, generate_hifi_data, generate_test_data, create_scalers, scale_data
from scipy.interpolate import griddata
from utility import GaussianKDE, unconcatenate_mu, unconcatenate_var
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

## Set device and seeds for numpy, torch and random
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cpu'

## Set directory name
direc_name = "2D_example"

## Set plotting parameters
title_fontsize = 9
axis_fontsize = 10
legend_fontsize = 8
linewidth = 1
scatter_size = 0.75

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

x_hifis = [x_hifi_1, x_hifi_2, x_hifi_3, x_hifi_4]
y_hifis = [y_hifi_1, y_hifi_2, y_hifi_3, y_hifi_4]

# Concatenate all the hifi data
x_hifi_data = torch.cat([x_hifi_1, x_hifi_2, x_hifi_3, x_hifi_4], dim=0)
y_hifi_data = torch.cat([y_hifi_1, y_hifi_2, y_hifi_3, y_hifi_4], dim=0)
x_lofi_data, y_lofi_data = generate_lofi_data(lofi_regions, lofi_regions_num_points)
x_test_data, y_test_data, x1_test_data_grid, x2_test_data_grid = generate_test_data(test_num=10000)
x_valid_data, y_valid_data, _, _ = generate_test_data(test_num=100)

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

# Store scaled data
x_hifis_scaled = [x_hifi_scaled_1, x_hifi_scaled_2, x_hifi_scaled_3, x_hifi_scaled_4]
y_hifis_scaled = [y_hifi_scaled_1, y_hifi_scaled_2, y_hifi_scaled_3, y_hifi_scaled_4]

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

# Unscaled train_sets
x_train_1_unscaled = x_hifi_1
x_train_2_unscaled = torch.concat([x_hifi_1, x_hifi_2], dim=0)
x_train_3_unscaled = torch.concat([x_train_2_unscaled, x_hifi_3], dim=0)
x_train_4_unscaled = torch.concat([x_train_3_unscaled, x_hifi_4], dim=0)
x_train_sets_unscaled = [x_train_1_unscaled, x_train_2_unscaled, x_train_3_unscaled, x_train_4_unscaled]

y_train_1_unscaled = y_hifi_1
y_train_2_unscaled = torch.concat([y_train_1_unscaled, y_hifi_2], dim=0)
y_train_3_unscaled = torch.concat([y_train_2_unscaled, y_hifi_3], dim=0)
y_train_4_unscaled = torch.concat([y_train_3_unscaled, y_hifi_4], dim=0)
y_train_sets_unscaled = [y_train_1_unscaled, y_train_2_unscaled, y_train_3_unscaled, y_train_4_unscaled]

# Place into tuples
train_sets = [(x_train_1, y_train_1), (x_train_2, y_train_2), (x_train_3, y_train_3), (x_train_4, y_train_4)]

# Concatenating the data
hifi_train_data = (x_hifi_scaled, y_hifi_scaled)
lofi_train_data = (x_lofi_scaled, y_lofi_scaled)
valid_data = (x_valid_scaled, y_valid_scaled)
test_data = (x_test_scaled, y_test_scaled)
x_Global_Scaler, y_Global_Scaler = scalers

#Plot the data in their sections
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for i in range(len(x_hifis)-2):
    # Plot original bins
    ax.scatter(x_hifis[i][:,0], x_hifis[i][:, 1], s=5, label=f'Update {i +1}')
    ax.set_xlabel('$x_1$', fontsize = axis_fontsize)
    ax.set_ylabel('$x_2$', fontsize = axis_fontsize)
    plt.legend(loc='lower right', fontsize=legend_fontsize)
plt.hlines(y=[0, 0.25, 0.5, 0.75, 1.0], xmin=0, xmax=1.0, colors=['black','black','black', 'black', 'black'], linestyles=['-', '-', '-', '-', '-'])
plt.vlines(x=[0, 0.25, 0.5, 0.75, 1.0], ymin=0, ymax=1.0, colors=['black','black','black', 'black', 'black'], linestyles=['-', '-', '-', '-', '-'])
plt.tight_layout()
#plt.show()
plt.savefig("data_plot.png", dpi=1000)

# Plot contour plots
for i in range(2):
    model = torch.load(r"".format(i), map_location=torch.device(device)) #\custom_1\train_set_{}\H_sto_single_prior.format(i) # place model name, something like this here
    mu, var = model(x_test_scaled)
    mu_H, mu_L, mu_D, H = unconcatenate_mu(mu)
    var_H, var_L, var_D = unconcatenate_var(var)

    #Unscale stuff
    mu_H = y_Global_Scaler.inverse_transform(mu_H)
    H = y_Global_Scaler.inverse_transform(H)
    var_H = y_Global_Scaler.inverse_transform_variance(var_H)

    # Detach tensors before conversion to NumPy
    mu_H = mu_H.detach()
    H = H.detach()
    var_H = var_H.detach()

    # Calculate MAE between test data and model outputs
    mae_Y_H = torch.mean(torch.abs(y_test_data - mu_H))
    mae_H = torch.mean(torch.abs(y_test_data - H))
    print("MSE $\mu_{Y_H}(x)$: ", mae_Y_H.item())
    print("MSE $H(x)$: ", mae_H.item())

    #Convert tensors to numpy
    x = x_test_data.cpu().numpy()
    y = y_test_data.cpu().numpy()
    mu_H_np = mu_H.cpu().numpy()
    var_H_np = var_H.cpu().numpy()

    # Define a grid to interpolate onto
    X1, X2 = x1_test_data_grid, x2_test_data_grid
    mu_H_grid = mu_H.reshape(100,100)
    y_test_grid = y.reshape(100,100)
    var_H_grid = var_H.reshape(100,100)

    # Plot surfaces
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    var_H_grid_clip = np.clip(var_H_grid, a_min=1e-10, a_max=None)

    # Calculate common scale range across both plots
    uncertainty = np.sqrt(var_H_grid_clip)
    error = np.abs(mu_H_grid - y_test_grid)

    common_vmin = 0.0
    common_vmax_1 = uncertainty.max()
    common_vmax_2 = error.max()
    common_vmaxs = [common_vmax_1, common_vmax_2]

    # Define the plots along with desired vmin and vmax
    plots = [(uncertainty, "Uncertainty $\sigma_{Y_H}(x)$"),(error, "Absolute Error $|\mu_{Y_H}(x) - y|$")]

    # Extract x1 and x2 for scatter
    x_hifi_np = x_train_sets_unscaled[i].cpu().numpy()  # if it's a torch tensor
    x1_hifi = x_hifi_np[:, 0]
    x2_hifi = x_hifi_np[:, 1]

    # Define plot data, titles, and common vmax values separately
    data_arrays = [uncertainty, error]
    titles = ["Uncertainty $\sigma_{Y_H}(x)$", "Absolute Error $|\mu_{Y_H}(x) - y|$"]
    common_vmaxs = [2.2, .85]
    common_vmin = 0.0  # Shared minimum

    # Plotting loop
    for ax, data, title, vmax in zip(axs, data_arrays, titles, common_vmaxs):
        im = ax.contourf(X1, X2, data, levels=100, cmap='viridis', vmin=common_vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)
        ax.scatter(x1_hifi, x2_hifi, c='red', s=16, label='HiFi Data', alpha=0.6, edgecolors='k')
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("uncertainty_error_plot_unscaled_{}.png".format(i), dpi=1000)
    #plt.show()

#Plot 2d contour plot of the KDE for hyperparameter tuning
H = torch.tensor([[0.003, 0], [0, 0.003]])
beta = torch.tensor(3.0)

# Random indexes for creating validation set from x_hifis_scaled [0]. Need to create generator
generator = torch.Generator(device=device).manual_seed(seed)
rand_idx = torch.randperm(x_hifis_scaled[0].shape[0], generator=generator)
# Want 20 of these values from rand_idx
rand_idx = rand_idx[:20]

# KDE plot
kde = GaussianKDE(x_hifis_scaled[0][rand_idx], H, beta=beta)
kde_output = 100 * kde.transformed_output(x_test_scaled)

# Prepare data
x = x_test_scaled[:, 0].cpu().numpy()
y = x_test_scaled[:, 1].cpu().numpy()
z = kde_output.cpu().numpy()

# Create 2D contour plot
fig, ax = plt.subplots(figsize=(5, 5))
contour = ax.tricontourf(x, y, z, levels=50, cmap='viridis')  # Filled contours
fig.colorbar(contour, ax=ax, label="KDE Output")

# Scatter original points (optional, can comment out)
ax.scatter(x_hifis_scaled[0][rand_idx,0], x_hifis_scaled[0][rand_idx, 1], c='red', s=5, edgecolors='k', linewidths=0.5, label='Hi-Fi Data')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('2D Contour of KDE Output')
ax.grid(True)
plt.tight_layout()