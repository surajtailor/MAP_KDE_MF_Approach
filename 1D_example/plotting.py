# %%
import torch
import numpy as np
from data_generation import generate_lofi_data, generate_hifi_data, generate_test_data, create_scalers, scale_data, lofi_function, function
import random
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utility import unconcatenate_mu, unconcatenate_var, GaussianKDE, beta_calculator

## Set device and seeds for numpy, torch and random
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cpu'

## Set plotting parameters
title_fontsize = 9
axis_fontsize = 10
legend_fontsize = 8
linewidth = 1
scatter_size = 0.75

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
x_hifi_data, y_hifi_data = generate_hifi_data(hifi_regions, hifi_regions_num_points)
x_test_data, y_test_data = generate_test_data(test_num=1000, start=0, end=1)
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

## Plot Hifi Lofi data along with generating functions
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
x = torch.linspace(0, 1, 1000)
y = function(x)
plt.plot(x, y, label= r'$y_{H}(x) = (x-\sqrt{2})y^2_L(x)$', linewidth = linewidth, color = 'orange')
y = lofi_function(x)
plt.plot(x, y, label= r'$y_{L}(x) = \sin(8\pi x)$', linewidth = linewidth, color = 'green')
ax.scatter(x_lofi_data.squeeze(), y_lofi_data.squeeze(), s=scatter_size, c='blue', label='Lo-Fi Data')
ax.scatter(x_hifi_data.squeeze(), y_hifi_data.squeeze(), s=scatter_size, c='red', label='Hi-Fi Data')
ax.set_xlabel('x', fontsize=axis_fontsize)
ax.set_ylabel('y', fontsize=axis_fontsize)
ax.legend(fontsize=legend_fontsize, loc='upper right')
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
ax.grid()
plt.savefig('1d_problem.png', dpi=1000, bbox_inches='tight')

# Plot changes in Sigma
beta = torch.tensor(3)
for band in [0.01, 0.1, 0.15, 0.2]:
    model = torch.load(r"".format(band))
    mu, var = model(x_test_scaled)
    mu_H, mu_L, mu_D, H = unconcatenate_mu(mu)
    var_H, var_L, var_D = unconcatenate_var(var)

    #Detach tensors before conversion to NumPy
    mu_H_np = mu_H.detach().numpy()
    var_H_np = var_H.detach().numpy()
    mu_L_np = mu_L.detach().numpy()
    mu_D_np = mu_D.detach().numpy()
    H_np = H.detach().numpy()

    # Create KDE from Hi-Fi data
    kde = GaussianKDE(x_hifi_scaled, band, beta=beta)
    kde_output = 100*kde.transformed_output(x_test_scaled).squeeze()

    # Plot mu_hat and var_hat
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(x_test_scaled.squeeze(), y_test_scaled.squeeze(), c='k', label='Ground Truth')
    ax.plot(x_test_scaled.squeeze(), H_np.squeeze(), c='g', label='$\hat{\mu}_{Y_H}(x)$')
    ax.plot(x_test_scaled.squeeze(), mu_L_np + mu_D_np, c='b', label='$\mu_{Y_L}(x)$ + $\mu_{Y_D}(x)$')
    ax.plot(x_test_scaled.squeeze(), mu_H_np,  c='r', label='$\mu_{Y_H}(x)$')
    ax.fill_between(x_test_scaled.squeeze(), mu_H_np - var_H_np, mu_H_np + var_H_np, alpha=0.3, color='r', label='$\mu_{Y_H}(x)\pm\sigma^2_{Y_H}(x)$')
    #ax.set_title('$\Sigma$ = {}'.format(band), fontsize=title_fontsize)
    ax.set_xlabel('x', fontsize=axis_fontsize)
    ax.set_ylabel('y', fontsize=axis_fontsize)
    ax.legend(fontsize=legend_fontsize,  loc='upper right')
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.grid()
    # Create inset axes
    inset_ax = inset_axes(ax, width="37%", height="37%",
                      loc='lower right',
                      bbox_to_anchor=(0.28, 0.095, 0.7, 0.7),
                      bbox_transform=ax.transAxes,
                      borderpad=0)
    # # Plot KDE values
    inset_ax.plot(x_test_scaled.squeeze(), kde_output.squeeze(), 'k-')
    inset_ax.set_xlabel('x', fontsize=6, labelpad=1)
    inset_ax.set_title(r'$KDE(x)$'.format(band), fontsize=6)
    inset_ax.tick_params(labelsize=6)
    inset_ax.grid()
    #plt.show()
    plt.savefig('1d_band_{}.png'.format(band), dpi=1000,bbox_inches='tight')

# Plot changes in kappa
beta = torch.tensor(3)
band = torch.tensor(0.01)
for kappa in [0.1, 1.0, 10.0, 100.0]:
    model = torch.load(r"".format(kappa))
    mu, var = model(x_test_scaled)
    mu_H, mu_L, mu_D, H = unconcatenate_mu(mu)
    var_H, var_L, var_D = unconcatenate_var(var)

    #Detach tensors before conversion to NumPy
    mu_H_np = mu_H.detach().numpy()
    var_H_np = var_H.detach().numpy()
    mu_L_np = mu_L.detach().numpy()
    mu_D_np = mu_D.detach().numpy()
    H_np = H.detach().numpy()

    # Create KDE from Hi-Fi data
    kde = GaussianKDE(x_hifi_scaled, band, beta=beta)
    kde_output = kappa*kde.transformed_output(x_test_scaled).squeeze()

    # Plot mu_hat and var_hat
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(x_test_scaled.squeeze(), y_test_scaled.squeeze(), c='k', label='Ground Truth')
    ax.plot(x_test_scaled.squeeze(), H_np.squeeze(), c='g', label='$H(x)$')
    ax.plot(x_test_scaled.squeeze(), mu_L_np + mu_D_np, c='b', label='$\mu_{Y_L}(x)$ + $\mu_{Y_D}(x)$')
    ax.plot(x_test_scaled.squeeze(), mu_H_np,  c='r', label='$\mu_{Y_H}(x)$')
    ax.fill_between(x_test_scaled.squeeze(), mu_H_np - var_H_np, mu_H_np + var_H_np, alpha=0.3, color='r', label='$\mu_{Y_H}(x)\pm\sigma^2_{Y_H}(x)$')
    #ax.set_title('$\kappa$ = {}'.format(kappa), fontsize=title_fontsize)
    ax.set_xlabel('x', fontsize=axis_fontsize)
    ax.set_ylabel('y', fontsize=axis_fontsize)
    ax.legend(fontsize=legend_fontsize,  loc='upper right')
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.grid()

    # Inset axes in bottom right (width/height as % of parent axes)
    #inset_ax = inset_axes(ax, width="30%", height="30%", loc='lower right')
    inset_ax = inset_axes(ax, width="37%", height="37%",
                      loc='lower right',
                      bbox_to_anchor=(0.28, 0.095, 0.7, 0.7),
                      bbox_transform=ax.transAxes,
                      borderpad=0)
    # # Plot KDE values
    inset_ax.plot(x_test_scaled.squeeze(), kde_output.squeeze(), 'k-')
    inset_ax.set_xlabel('x', fontsize=6, labelpad=1)
    inset_ax.set_title(r'$KDE(x)$', fontsize=6)
    inset_ax.tick_params(labelsize=6)
    inset_ax.grid()
    #plt.show()
    plt.savefig('1d_kappa_{}.png'.format(kappa), dpi=1000,bbox_inches='tight')

# Plot changes in beta_0
beta = torch.tensor(3)
band = torch.tensor(0.01)
for beta_0 in [0.2, 0.8, 1.4, 2.0]:
    model = torch.load(r"".format(beta_0))
    mu, var = model(x_test_scaled)
    mu_H, mu_L, mu_D, H = unconcatenate_mu(mu)
    var_H, var_L, var_D = unconcatenate_var(var)

    #Detach tensors before conversion to NumPy
    mu_H_np = mu_H.detach().numpy()
    var_H_np = var_H.detach().numpy()
    H_np = H.detach().numpy()
    mu_L_np = mu_L.detach().numpy()
    mu_D_np = mu_D.detach().numpy()

    # Create KDE from Hi-Fi data
    kde = GaussianKDE(x_hifi_scaled, band, beta=beta)
    kde_output = 100*kde.transformed_output(x_test_scaled).squeeze()

    # Plot mu_hat and var_hat
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(x_test_scaled.squeeze(), y_test_scaled.squeeze(), c='k', label='Ground Truth')
    ax.plot(x_test_scaled.squeeze(), H_np.squeeze(), c='g', label='$H(x)$')
    ax.plot(x_test_scaled.squeeze(), mu_L_np + mu_D_np, c='b', label='$\mu_{Y_L}(x)$ + $\mu_{Y_D}(x)$')
    ax.plot(x_test_scaled.squeeze(), mu_H_np,  c='r', label='$\mu_{Y_H}(x)$')
    ax.fill_between(x_test_scaled.squeeze(), mu_H_np - var_H_np, mu_H_np + var_H_np, alpha=0.3, color='r', label='$\mu_{Y_H}(x)\pm\sigma^2_{Y_H}(x)$')
    #ax.set_title(r'$\beta_0$ = {}'.format(beta_0), fontsize=title_fontsize)
    ax.set_xlabel('x', fontsize=axis_fontsize)
    ax.set_ylabel('y', fontsize=axis_fontsize)
    ax.legend(fontsize=legend_fontsize,  loc='upper right')
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.grid()

    # Inset axes in bottom right (width/height as % of parent axes)
    #inset_ax = inset_axes(ax, width="30%", height="30%", loc='lower right')
    inset_ax = inset_axes(ax, width="37%", height="37%",
                      loc='lower right',
                      bbox_to_anchor=(0.28, 0.095, 0.7, 0.7),
                      bbox_transform=ax.transAxes,
                      borderpad=0)
    # # Plot KDE values
    inset_ax.plot(x_test_scaled.squeeze(), kde_output.squeeze(), 'k-')
    inset_ax.set_xlabel('x', fontsize=6, labelpad=1)
    inset_ax.set_title(r'$KDE(x)$', fontsize=6)
    inset_ax.tick_params(labelsize=6)
    inset_ax.grid()
    #plt.show()
    plt.savefig('1d_beta_0_{}.png'.format(beta_0), dpi=1000,bbox_inches='tight')