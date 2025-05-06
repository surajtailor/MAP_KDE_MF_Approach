import torch
import torch.nn as nn
from scipy.interpolate import CloughTocher2DInterpolator as CT
import numpy as np

class NetAnyFunctional2D(nn.Module):
    def __init__(self,
                 input_size=2,
                 hidden_size=10,
                 num_layers=5,
                 training=True,
                 device=torch.device('cuda'),
                 functional='polynomial',
                 poly_degree=2,
                 fourier_degree=None,
                 poly_l_degree=None,
                 custom = False,
                 interpolate_H = False,
                 activation_func='elu'):
        super(NetAnyFunctional2D, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.training = training
        self.functional = functional
        self.poly_degree = poly_degree
        self.fourier_degree = fourier_degree
        self.poly_l_degree = poly_l_degree
        self.custom = custom
        self.num_layers = num_layers
        self.interpolate_H = interpolate_H

        # Set activation function
        self.activation = self._get_activation_function(activation_func)

        # Define the networks
        self.mu_H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)
        self.var_H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

        self.mu_L = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)
        self.var_L = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

        if self.interpolate_H:
            self.f= lambda x: np.zeros((x.shape[0],), dtype=np.float32)  # fallback if interpolation not created
        else:
            self.H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)

        # Initialize parameters for different functionals
        self.D_params = []
        self._initialize_functional_params()
        self.var_D = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

    def _get_activation_function(self, activation_func):
        """Returns the specified activation function."""
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'leakyrelu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'hat': HatActivation()
        }
        return activations.get(activation_func, nn.ELU())

    def _build_network(self, input_size, hidden_size, num_layers, activation, final_output_size=None, final_activation=None):
        """Builds a sequential network with a specified number of layers and activations."""
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        if final_output_size:
            layers.append(nn.Linear(hidden_size, final_output_size))
        if final_activation:
            layers.append(final_activation)
        return nn.Sequential(*layers)

    def _initialize_functional_params(self):
        """Initializes parameters based on the selected functional type."""
        if self.functional == 'polynomial':
            for i in range(self.poly_degree*2 + 1):
                param = nn.Parameter(torch.randn(1))
                setattr(self, f'A{i}', param)
                self.D_params.append(param)
        if self.functional == 'custom_1' or self.functional == 'custom_2':
            param_names = ['A', 'B', 'C']  # Could easily extend this
            for name in param_names:
                param = nn.Parameter(torch.randn(1))
                setattr(self, name, param)
                self.D_params.append(param)
        if self.functional == 'custom_3':
            param_names = ['A', 'B']
            for name in param_names:
                param = nn.Parameter(torch.randn(1))
                setattr(self, name, param)
                self.D_params.append(param)


    def _compute_D(self, input, mu_L):
        D = torch.zeros(input.size(0), device=input.device, dtype=input.dtype)
        L_det = mu_L.detach().squeeze()

        if self.functional == 'polynomial':
            for i in range(self.poly_degree + 1):
                D += getattr(self, f'A{i}') * input[:, 0] ** i
            for i in range(self.poly_degree + 1, 2 * self.poly_degree + 1):
                D += getattr(self, f'A{i}') * input[:, 1] ** (i - self.poly_degree)
        elif self.functional == 'custom_1':
            D += getattr(self, 'A') * L_det
            D += getattr(self, 'B') * input[:, 0]
            D += getattr(self, 'C')
        elif self.functional == 'custom_2':
            D += getattr(self, 'A') * L_det
            D += getattr(self, 'B') * input[:, 0]*input[:, 1]
            D += getattr(self, 'C')
        elif self.functional == 'custom_3':
            D += getattr(self, 'A') * L_det
            D += getattr(self, 'B')
        else:
            raise ValueError(f"Unknown functional type: {self.functional}")
        return D

    def _create_interpolate_H(self, hifi_train_data):
        x_hifi_data, y_hifi_data = hifi_train_data

        self.x1_data = x_hifi_data[:, 0]
        self.x2_data = x_hifi_data[:, 1]
        self.y_data = y_hifi_data
        self.f = CT(x_hifi_data, y_hifi_data)
        self.created_interpolate_H = True

    def _final_f(self, input):
        xx, yy = input[:, 0], input[:, 1]
        # evaluate the CT interpolator. Out-of-bounds values are nan.
        zz = self.f(xx, yy)
        nans = np.isnan(zz)
        if nans.any():
            # for each nan point, find its nearest neighbor
            inds = np.argmin(
                (self.x1_data[:, None] - xx[nans]) ** 2 +
                (self.x2_data[:, None] - yy[nans]) ** 2
                , axis=0)
            # ... and use its value
            zz[nans] = self.y_data[inds]
        return zz

    def forward(self, input):
        # Check to determine if interpolation function has been created only if interpolate_H is True
        if self.interpolate_H and not self.created_interpolate_H:
            raise ValueError("Interpolation function 'f' has not been created. Please call _create_interpolate_H method first.")

        # Compute mu_H, var_H, mu_L, var_L, mu_D, var_D
        var_H = self.var_H(input)
        mu_H = self.mu_H(input)

        var_L = self.var_L(input)
        mu_L = self.mu_L(input)

        mu_D = self._compute_D(input, mu_L).reshape((-1,1))
        var_D = self.var_D(input)

        # Interpolate H if needed
        if self.interpolate_H:
            input_cpu = input.detach().cpu().numpy()
            interpolated_vals = self._final_f(input_cpu)
            H = torch.from_numpy(interpolated_vals).float().unsqueeze(1).to(input.device)
        else:
            H = self.H(input)

        mu = torch.cat((mu_H, mu_L, mu_D, H), dim=1)
        var = torch.cat((var_H, var_L, var_D), dim=1)
        return mu, var

class HatActivation(nn.Module):
    def __init__(self):
        super(HatActivation, self).__init__()

    def forward(self, x):
        return torch.where((x >= 0) & (x < 1), x, torch.where((x >= 1) & (x < 2), 2 - x, torch.zeros_like(x)))

class NetNNFunctional(nn.Module):
    def __init__(self,
                 input_size=1,
                 hidden_size=10,
                 num_layers=5,
                 training=True,
                 device=torch.device('cuda'),
                 activation_func='elu'):
        super(NetNNFunctional, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.training = training
        self.num_layers = num_layers

        # Set activation function
        self.activation = self._get_activation_function(activation_func)

        # Define the networks
        self.mu_H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)
        self.var_H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

        self.mu_L = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)
        self.var_L = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

        self.H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)

        self.mu_D = self._build_network(2, self.hidden_size, self.num_layers, self.activation, final_output_size=1)
        self.var_D = self._build_network(2, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

    def _get_activation_function(self, activation_func):
        """Returns the specified activation function."""
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'leakyrelu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'hat': HatActivation()
        }
        return activations.get(activation_func, nn.ELU())

    def _build_network(self, input_size, hidden_size, num_layers, activation, final_output_size=None, final_activation=None):
        """Builds a sequential network with a specified number of layers and activations."""
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        if final_output_size:
            layers.append(nn.Linear(hidden_size, final_output_size))
        if final_activation:
            layers.append(final_activation)
        return nn.Sequential(*layers)

    def forward(self, input):
        var_H = self.var_H(input)
        mu_H = self.mu_H(input)

        var_L = self.var_L(input)
        mu_L = self.mu_L(input)

        # Concatenate input and mu_L for mu_D input in the second dimension
        mu_L_det = mu_L.detach()
        input_concat = torch.cat((input, mu_L_det), dim=1)
        mu_D = self.mu_D(input_concat)
        var_D = self.var_D(input_concat)

        H = self.H(input)

        mu = torch.cat((mu_H, mu_L, mu_D, H), dim=1)
        var = torch.cat((var_H, var_L, var_D), dim=1)
        return mu, var

class NetAnyFunctional2DHNN(nn.Module):
    def __init__(self,
                 input_size=2,
                 hidden_size=10,
                 num_layers=5,
                 training=True,
                 device=torch.device('cuda'),
                 functional='polynomial',
                 poly_degree=2,
                 fourier_degree=None,
                 poly_l_degree=None,
                 custom = False,
                 interpolate_H = False,
                 activation_func='elu'):
        super(NetAnyFunctional2D, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.training = training
        self.functional = functional
        self.poly_degree = poly_degree
        self.fourier_degree = fourier_degree
        self.poly_l_degree = poly_l_degree
        self.custom = custom
        self.num_layers = num_layers
        self.interpolate_H = interpolate_H

        # Set activation function
        self.activation = self._get_activation_function(activation_func)

        # Define the networks
        self.mu_H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)
        self.var_H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

        self.mu_L = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)
        self.var_L = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

        if self.interpolate_H:
            self.f= lambda x: np.zeros((x.shape[0],), dtype=np.float32)  # fallback if interpolation not created
        else:
            self.H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)

        # Initialize parameters for different functionals
        self.D_params = []
        self._initialize_functional_params()
        self.var_D = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

    def _get_activation_function(self, activation_func):
        """Returns the specified activation function."""
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'leakyrelu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'hat': HatActivation()
        }
        return activations.get(activation_func, nn.ELU())

    def _build_network(self, input_size, hidden_size, num_layers, activation, final_output_size=None, final_activation=None):
        """Builds a sequential network with a specified number of layers and activations."""
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        if final_output_size:
            layers.append(nn.Linear(hidden_size, final_output_size))
        if final_activation:
            layers.append(final_activation)
        return nn.Sequential(*layers)

    def _initialize_functional_params(self):
        """Initializes parameters based on the selected functional type."""
        if self.functional == 'polynomial':
            for i in range(self.poly_degree*2 + 1):
                param = nn.Parameter(torch.randn(1))
                setattr(self, f'A{i}', param)
                self.D_params.append(param)
        if self.functional == 'custom_1' or self.functional == 'custom_2':
            param_names = ['A', 'B', 'C']
            initial = [0.3,0.005,2.3]
            for name, init in zip(param_names, initial):
                param = nn.Parameter(init)
                setattr(self, name, param)
                self.D_params.append(param)

    def _compute_D(self, input, mu_L):
        D = torch.zeros(input.size(0), device=input.device, dtype=input.dtype)
        L_det = mu_L.detach().squeeze()

        if self.functional == 'polynomial':
            for i in range(self.poly_degree + 1):
                D += getattr(self, f'A{i}') * input[:, 0] ** i
            for i in range(self.poly_degree + 1, 2 * self.poly_degree + 1):
                D += getattr(self, f'A{i}') * input[:, 1] ** (i - self.poly_degree)
        elif self.functional == 'custom_1':
            D += getattr(self, 'A') * L_det
            D += getattr(self, 'B') * input[:, 0]
            D += getattr(self, 'C')
        elif self.functional == 'custom_2':
            D += getattr(self, 'A') * L_det
            D += getattr(self, 'B') * input[:, 0]*input[:, 1]
            D += getattr(self, 'C')
        else:
            raise ValueError(f"Unknown functional type: {self.functional}")
        return D

    def _create_interpolate_H(self, hifi_train_data):
        x_hifi_data, y_hifi_data = hifi_train_data

        self.x1_data = x_hifi_data[:, 0]
        self.x2_data = x_hifi_data[:, 1]
        self.y_data = y_hifi_data
        self.f = CT(x_hifi_data, y_hifi_data)
        self.created_interpolate_H = True

    def _final_f(self, input):
        xx, yy = input[:, 0], input[:, 1]
        # evaluate the CT interpolator. Out-of-bounds values are nan.
        zz = self.f(xx, yy)
        nans = np.isnan(zz)
        if nans.any():
            # for each nan point, find its nearest neighbor
            inds = np.argmin(
                (self.x1_data[:, None] - xx[nans]) ** 2 +
                (self.x2_data[:, None] - yy[nans]) ** 2
                , axis=0)
            # ... and use its value
            zz[nans] = self.y_data[inds]
        return zz

    def forward(self, input):
        # Check to determine if interpolation function has been created only if interpolate_H is True
        if self.interpolate_H and not self.created_interpolate_H:
            raise ValueError("Interpolation function 'f' has not been created. Please call _create_interpolate_H method first.")

        # Compute mu_H, var_H, mu_L, var_L, mu_D, var_D
        var_H = self.var_H(input)
        mu_H = self.mu_H(input)

        var_L = self.var_L(input)
        mu_L = self.mu_L(input)

        mu_D = self._compute_D(input, mu_L).reshape((-1,1))
        var_D = self.var_D(input)

        # Interpolate H if needed
        if self.interpolate_H:
            input_cpu = input.detach().cpu().numpy()
            interpolated_vals = self._final_f(input_cpu)
            H = torch.from_numpy(interpolated_vals).float().unsqueeze(1).to(input.device)
        else:
            H = self.H(input)

        mu = torch.cat((mu_H, mu_L, mu_D, H), dim=1)
        var = torch.cat((var_H, var_L, var_D), dim=1)
        return mu, var