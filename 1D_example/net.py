import torch
import torch.nn as nn
import scipy.interpolate as interpolate
class HatActivation(nn.Module):
    def __init__(self):
        super(HatActivation, self).__init__()

    def forward(self, x):
        return torch.where((x >= 0) & (x < 1), x, torch.where((x >= 1) & (x < 2), 2 - x, torch.zeros_like(x)))
class NetAnyFunctional(nn.Module):
    def __init__(self,
                 input_size=1,
                 hidden_size=10,
                 num_layers=5,
                 training=True,
                 device=torch.device('cuda'),
                 functional='polynomial',
                 poly_degree=2,
                 fourier_degree=None,
                 poly_l_degree=None,
                 interpolate_H = False,
                 activation_func='elu'):
        super(NetAnyFunctional, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.training = training
        self.functional = functional
        self.poly_degree = poly_degree
        self.fourier_degree = fourier_degree
        self.poly_l_degree = poly_l_degree
        self.num_layers = num_layers
        self.interpolate_H = interpolate_H

        # Set activation function
        self.activation = self._get_activation_function(activation_func)

        # Define the networks
        self.mu_H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)
        self.var_H = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

        self.mu_L = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1)
        self.var_L = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=1, final_activation=nn.Softplus())

        if self.interpolate_H == True:
            self.f = lambda x: torch.zeros_like(x)
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
            for i in range(self.poly_degree + 1):
                param = nn.Parameter(torch.randn(1))
                setattr(self, f'A{i}', param)
                self.D_params.append(param)
        elif self.functional == 'periodic_fourier':
            self.A0 = nn.Parameter(torch.randn(1))
            self.D_params.append(self.A0)
            for i in range(self.fourier_degree):
                param_B = nn.Parameter(torch.randn(1))
                param_C = nn.Parameter(torch.randn(1))
                setattr(self, f'B{i}', param_B)
                setattr(self, f'C{i}', param_C)
                self.D_params.extend([param_B, param_C])
        elif self.functional == 'periodic':
            for name in ['A0', 'A1', 'A2', 'A3']:
                param = nn.Parameter(torch.randn(1))
                setattr(self, name, param)
                self.D_params.append(param)
        elif self.functional == 'polynomial_on_l':
            for i in range(self.poly_l_degree + 1):
                param = nn.Parameter(torch.randn(1))
                setattr(self, f'A{i}', param)
                self.D_params.append(param)
        elif self.functional == 'polynomial_periodic':
            for i in range(self.poly_degree):
                param = nn.Parameter(torch.randn(1))
                setattr(self, f'A{i}', param)
                self.D_params.append(param)
            for name in ['B0', 'B1', 'B2', 'B3']:
                param = nn.Parameter(torch.randn(1))
                setattr(self, name, param)
                self.D_params.append(param)
        elif self.functional == 'polynomial_polynomial_on_l':
            self.A0 = nn.Parameter(torch.randn(1))
            self.D_params.append(self.A0)
            for i in range(self.poly_degree):
                param_B = nn.Parameter(torch.randn(1))
                setattr(self, f'B{i}', param_B)
                self.D_params.append(param_B)
            for i in range(self.poly_l_degree):
                param_C = nn.Parameter(torch.randn(1))
                setattr(self, f'C{i}', param_C)
                self.D_params.append(param_C)
    def _compute_D(self, input, mu_L):
        D = 0
        L_det = mu_L.detach()
        if self.functional == 'polynomial':
            for i in range(self.poly_degree + 1):
                D += getattr(self, f'A{i}') * input ** i
        elif self.functional == 'periodic_fourier':
            D += self.A0 * input ** 0
            for i in range(self.fourier_degree):
                D += getattr(self, f'B{i}') * torch.sin((i + 1) * input)
                D += getattr(self, f'C{i}') * torch.cos((i + 1) * input)
        elif self.functional == 'periodic':
            D = self.A0 + self.A1 * torch.sin(self.A2 * input + self.A3)
        elif self.functional == 'polynomial_on_l':
            for i in range(self.poly_l_degree + 1):
                D += getattr(self, f'A{i}') * L_det ** i
        elif self.functional == 'polynomial_periodic':
            for i in range(self.poly_degree):
                D += getattr(self, f'A{i}') * input ** (i+1)
            D += self.B0 + self.B1 * torch.sin(self.B2 * input + self.B3)
        elif self.functional == 'polynomial_polynomial_on_l':
            D += self.A0 * input ** 0
            for i in range(self.poly_degree):
                D += getattr(self, f'B{i}') * input ** (i+1)
            for i in range(self.poly_l_degree):
                D += getattr(self, f'C{i}') * L_det ** (i+1)
        return D

    def _create_interpolate_H(self, hifi_train_data):
        x_hifi_data, y_hifi_data = hifi_train_data
        self.f = interpolate.interp1d(
            x_hifi_data.squeeze().cpu().numpy(),
            y_hifi_data.squeeze().cpu().numpy(),
            fill_value="extrapolate"
        )
        self.created_interpolate_H = True

    def forward(self, input):
        # Check to determine if interpolation function has been created only if interpolate_H is True
        if self.interpolate_H and not self.created_interpolate_H:
            raise ValueError("Interpolation function 'f' has not been created. Please call _create_interpolate_H method first.")

        # Compute mu_H, var_H, mu_L, var_L, mu_D, var_D
        var_H = self.var_H(input)
        mu_H = self.mu_H(input)

        var_L = self.var_L(input)
        mu_L = self.mu_L(input)

        mu_D = self._compute_D(input, mu_L)
        var_D = self.var_D(input)

        # Interpolate H if needed
        if self.interpolate_H:
            input_cpu = input.squeeze().detach().cpu().numpy()
            interpolated_vals = self.f(input_cpu)
            H = torch.from_numpy(interpolated_vals).float().unsqueeze(1).to(input.device)
        else:
            H = self.H(input)

        mu = torch.cat((mu_H, mu_L, mu_D, H), dim=1)
        var = torch.cat((var_H, var_L, var_D), dim=1)
        return mu, var