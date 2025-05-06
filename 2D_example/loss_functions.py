import torch
from utility import unconcatenate_mu, unconcatenate_var

def loss_fn(mu, var, targets, indices = None, hyperparameters = None, case = 'L_sto', tau= 0.5, tau_adjustment = False):
    mu_H, mu_L, mu_D, H = unconcatenate_mu(mu)
    var_H, var_L, var_D = unconcatenate_var(var)

    if case == 'L_sto': # Lofi_Stochastic_Case
        L_log_likelihood = -(1 / 2) * torch.log(var_L) - (1 / 2) * (mu_L - targets) ** 2 / (var_L)
        if tau_adjustment == True:
            L_log_likelihood = var_L.detach() ** (tau) * L_log_likelihood
        return -torch.mean(L_log_likelihood)

    if case == 'L':
        se_L = (mu_L - targets) ** 2
        return torch.mean(se_L)

    if case == 'H':
        se_H = (H - targets) ** 2
        return  torch.mean(se_H)

    if case == 'D_sto': #Evaluate only at areas where H is available i.e. train on H input data
        D = (H.detach() - mu_L.detach())
        D_log_likelihood = -(1 / 2) * torch.log(var_D) - (1 / 2) * (mu_D - D) ** 2 / (var_D)
        if tau_adjustment == True:
            D_log_likelihood = var_D.detach() ** (tau) * D_log_likelihood
        return -torch.mean(D_log_likelihood)

    if case == 'D_sto_NN': #Evaluate only at areas where H is available i.e. train on H input data
        D_log_likelihood = -(1 / 2) * torch.log(var_D) - (1 / 2) * (mu_D - H.detach()) ** 2 / (var_D)
        if tau_adjustment == True:
            D_log_likelihood = var_D.detach() ** (tau) * D_log_likelihood
        return -torch.mean(D_log_likelihood)

    if case == 'D':
        D = (H.detach() - mu_L.detach())
        se_D = (mu_D - D) ** 2
        return torch.mean(se_D)

    if case == 'D_NN':
        se_D = (mu_D - H.detach()) ** 2
        return torch.mean(se_D)

    if case == 'H_sto':
        n_H, sig_prior, sig_mismatch, alpha_0, beta_0 = hyperparameters
        n_H = n_H[indices]
        H_log_likelihood = -(n_H / 2) * torch.log(var_H) - (n_H / 2) * (mu_H - H.detach()) ** 2 / (var_H)
        mu_H_mismatch = -(1/2)*(mu_H - (mu_L.detach()+ mu_D.detach()))**2/(sig_mismatch)
        mu_H_prior = -(1/2)*(mu_H - mu_L.detach())**2/(sig_prior)
        H_log_var_prior = -(alpha_0 + 1) * torch.log(var_H) - (beta_0) / (var_H)
        posterior_log = H_log_likelihood + mu_H_mismatch + mu_H_prior + H_log_var_prior
        return  -torch.mean(posterior_log)

    if case == 'H_sto_single_prior': # Change from mu_D to mu_L + mu_D
        n_H, sig_prior, sig_mismatch, alpha_0, beta_0 = hyperparameters
        n_H = n_H[indices]
        H_log_likelihood = -(n_H / 2) * torch.log(var_H) - (n_H / 2) * (mu_H - H.detach()) ** 2 / (var_H)
        mu_H_mismatch = -(1/2)*(mu_H - (mu_L.detach() + mu_D.detach()))**2/(sig_mismatch)
        H_log_var_prior = -(alpha_0 + 1) * torch.log(var_H) - (beta_0) / (var_H)
        posterior_log = H_log_likelihood + mu_H_mismatch + H_log_var_prior
        return  -torch.mean(posterior_log)

def loss_mse(mu, targets):
    mu_H, mu_L, mu_D, H = unconcatenate_mu(mu)
    se_H = (mu_H - targets) ** 2
    return torch.mean(se_H)

def logmeanexp(inputs, dim=0):
    input_max = inputs.max(dim=dim)[0]
    return (inputs - input_max).exp().mean(dim=dim).log() + input_max