import torch
import numpy as np
from data_generation import generate_lofi_data, generate_hifi_data, generate_test_data, create_scalers, scale_data, function, lofi_function
from utility import split_into_domain_bins_2D, unconcatenate_mu, unconcatenate_var, GaussianKDE
import random
import matplotlib.pyplot as plt

## Set device and seeds for numpy, torch and random
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cpu'
scale_data_switch = True

## Set directory name
direc_name = "2D_example"

## Set plotting parameters
title_fontsize = 9
axis_fontsize = 10
legend_fontsize = 8
linewidth = 1
scatter_size = 0.75

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
lofi_train_data = (x_lofi_scaled, y_lofi_scaled)
valid_data = (x_valid_scaled, y_valid_scaled)
x_Global_Scaler, y_Global_Scaler = scalers

# Concatenating the data
hifi_train_data = (x_hifi_scaled, y_hifi_scaled)
lofi_train_data = (x_lofi_scaled, y_lofi_scaled)
valid_data = (x_valid_scaled, y_valid_scaled)
test_data = (x_test_scaled, y_test_scaled)
x_Global_Scaler, y_Global_Scaler = scalers

bins, bins_scaled = split_into_domain_bins_2D(x_hifi_data, y_hifi_data, x_hifi_scaled, y_hifi_scaled, 3, [0, 1, 0, 1], device="cpu")

x_lofi_data, y_lofi_data = generate_lofi_data(lofi_regions, lofi_regions_num_points)
x_hifi_data, y_hifi_data = generate_hifi_data(hifi_regions, hifi_regions_num_points)
x_test_data, y_test_data = generate_test_data(test_num=1000)
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


## Create all training sets
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

# Create validation sets
x_valid_1 = bins_scaled[0][0]
x_valid_2 = torch.concat([bins_scaled[0][0], bins_scaled[6][0]], dim=0)
x_valid_sets = [x_valid_1, x_valid_2]

y_valid_1 = bins_scaled[0][1]
y_valid_2 = torch.concat([bins_scaled[0][1], bins_scaled[6][1]], dim=0)
y_valid_sets = [y_valid_1, y_valid_2]

## Plotting ground truth and Lofi function
# Show ground truth data. Scale data for plot visualisation
x1 = torch.linspace(0, 1, 1000)
x2 = torch.linspace(0, 1, 1000)
X1, X2 = torch.meshgrid(x1, x2)
X = torch.vstack((X1.flatten(), X2.flatten())).T
Y_hifi = function(X)
Y_lofi = lofi_function(X)

# Scale the 2D input grid
X_scaled = x_Global_Scaler.transform(X)  # X has shape (1_000_000, 2)

# Unpack scaled grid into 2D plot format
X1_scaled = X_scaled[:, 0].reshape(X1.shape)
X2_scaled = X_scaled[:, 1].reshape(X2.shape)

# Scale the outputs
Y_hifi_scaled = y_Global_Scaler.transform(Y_hifi).reshape(X1.shape)
Y_lofi_scaled = y_Global_Scaler.transform(Y_lofi).reshape(X1.shape)

# Plot the ground truth
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y_hifi.reshape(X1.shape), alpha=1, cmap='viridis', label = '$y_H = (x_1 - 1)^2 + (2x_2^2 - x_1)^2$')
ax.set_xlabel('$x_1$', fontsize=title_fontsize)
ax.set_ylabel('$x_2$', fontsize=title_fontsize)
ax.set_zlabel('$y$', fontsize=title_fontsize)
#ax.set_title('Ground Truth Plot', fontsize=title_fontsize)
ax.grid()
plt.legend(loc='center left', fontsize=legend_fontsize)
plt.savefig('2d_ground_truth_plot.png', dpi=1000, bbox_inches='tight')

# Plot the lofi function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y_lofi.reshape(X1.shape), alpha=1, cmap='plasma', label='$y_L = 0.8 * y_H - 0.4 * x_1 * x_2 - 50$')
ax.set_xlabel('$x_1$', fontsize=title_fontsize)
ax.set_ylabel('$x_2$', fontsize=title_fontsize)
ax.set_zlabel('$y$', fontsize=title_fontsize)
#ax.set_title('Lo-Fi Function Plot', fontsize=title_fontsize)
ax.grid()
plt.legend(loc='center left', fontsize=legend_fontsize)
plt.savefig('2d_lofi_function_plot.png', dpi=1000, bbox_inches='tight')

## Evaluating the MSE of the Models
mse_H_store = []
mse_mu_H_store = []
for i in range(7):
    model = torch.load(r"".format(i), map_location=torch.device(device))
    mu, var = model(x_test_scaled)
    mu_H, mu_L, mu_D, H = unconcatenate_mu(mu)
    var_H, var_L, var_D = unconcatenate_var(var)

    #Detach tensors before conversion to NumPy
    mu_H_np = mu_H.detach()
    var_H_np = var_H.detach()
    H_np = H.detach()
    mu_L_np = mu_L.detach()
    mu_D_np = mu_D.detach()

    mu_H_np = y_Global_Scaler.inverse_transform(mu_H_np)
    mu_L_np = y_Global_Scaler.inverse_transform(mu_L_np)
    mu_D_np = y_Global_Scaler.inverse_transform_mu_D(mu_D_np)
    var_H_np = y_Global_Scaler.inverse_transform_variance(var_H_np)

    # Calculate MSE between test data and model outputs
    mse = torch.mean((y_test_data - mu_H_np) ** 2)
    mse_H = torch.mean((y_test_data - H_np) ** 2)
    mse_L = torch.mean((y_test_data - mu_L_np) ** 2)

    # Store the MSE Values
    mse_H_store.append(mse_H)
    mse_mu_H_store.append(mse)

# Print the MSE values
print("MSE for mu_hat_H model: ", mse_H_store)
print("MSE for mu_H model: ", mse_mu_H_store)

## Plot the KDE values on the subsets
# First and second hyper tuning
H_1 = torch.tensor([[0.005, 0], [0, 0.005]])
H_2 = torch.tensor([[0.02, 0], [0, 0.005]])
H_store = [H_1, H_2]

# Create figure with 1x2 grid for subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Loop through each H value and plot on separate axes
for i, H in enumerate(H_store):
    beta = torch.tensor(5.0) # Fixed for both KDEs
    kde = GaussianKDE(x_valid_sets[i], H, beta=beta)
    kde_output = 100 * kde.transformed_output(x_test_scaled)

    # Prepare data
    x = x_test_scaled[:, 0].cpu().numpy()
    y = x_test_scaled[:, 1].cpu().numpy()
    z = kde_output.cpu().numpy()

    # Plot on the ith axis (i.e., axes[0] and axes[1])
    contour = axes[i].tricontourf(x, y, z, levels=50, cmap='viridis')  # Filled contours
    axes[i].set_xlabel('$x_1$',     fontsize=axis_fontsize)
    axes[i].set_ylabel('$x_2$', fontsize=axis_fontsize)

    # Scatter original points (optional)
    #axes[i].scatter(x, y, c='white', s=5, edgecolors='k', linewidths=0.5)
    axes[i].grid(True)

# Add a single color bar that applies to both axes
fig.colorbar(contour, ax=axes, label="KDE Output", orientation='vertical')
# Adjust layout for better spacing
plt.savefig("2d_kde_output.png", dpi=1000)

## Plot the Hi-Fi data and sections
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for i in [0,1,2,3,5,6,7,8]:
    bin = bins[i][0].cpu().numpy()
    bin_scaled = bins_scaled[i][0].cpu().numpy()
        # Plot original bins
    ax.scatter(bin[:, 0], bin[:, 1], s=10, label=f'Section {i}')
    ax.set_xlabel('$x_1$', fontsize = axis_fontsize)
    ax.set_ylabel('$x_2$', fontsize = axis_fontsize)
    #plt.legend(loc='upper right', fontsize=legend_fontsize)
plt.tight_layout()
#plt.grid()
plt.savefig("2d_data_plot_no_legend.png", dpi=1000)