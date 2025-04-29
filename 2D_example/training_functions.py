import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR
from loss_functions import loss_mse, loss_fn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from itertools import chain
from utility import IndexedDataset

def scheduler_case(scheduler_type, optimizer, initial_lr, num_epochs, final_lr = 1e-4, steps = 3):
    if scheduler_type == "ExponentialLR":
        gamma_value = (final_lr/initial_lr)**(1/num_epochs)
        return ExponentialLR(optimizer, gamma=gamma_value)

    if scheduler_type == "StepLR":
        step_size = num_epochs/steps
        gamma_value = (final_lr/initial_lr)**(1/steps)
        return StepLR(optimizer, step_size, gamma=gamma_value)
    return None

def trainer(model, train_data, valid_data, experiment_name, model_save_name, hyperparameters = None, run_name = None, device=torch.device('cpu'), n_iter=10000, learning_rate=1e-3,scheduler_type=None, save_freq=100, weight_decay=0, valid_freq=1, tau = 0.5, tau_adjustment = False, batch_size = 40, num_workers = 0, case = 'L_sto'):
    mlflow.set_experiment(experiment_name)  # Set the name of your experiment
    if case == 'L_sto':
        optimizer = optim.Adam(chain(model.mu_L.parameters(), model.var_L.parameters()), lr=learning_rate, weight_decay=weight_decay)
    if case == 'L':
        optimizer = optim.Adam(chain(model.mu_L.parameters()), lr=learning_rate, weight_decay=weight_decay)
    if case == 'H':
        optimizer = optim.Adam(chain(model.H.parameters()), lr=learning_rate, weight_decay=weight_decay)
    if case == 'D_sto':
        optimizer = optim.Adam(chain(model.D_params, model.var_D.parameters()), lr=learning_rate, weight_decay=weight_decay)
    if case == 'D':
        optimizer = optim.Adam(chain(model.D_params), lr=learning_rate, weight_decay=weight_decay)
    if case == 'H_sto' or case == 'H_sto_single_prior':
        optimizer = optim.Adam(chain(model.mu_H.parameters(), model.var_H.parameters()), lr=learning_rate, weight_decay=weight_decay)
    if case == 'D_NN':
        optimizer = optim.Adam(chain(model.mu_D.parameters()), lr=learning_rate, weight_decay=weight_decay)
    if case == 'D_sto_NN':
        optimizer = optim.Adam(chain(model.mu_D.parameters(), model.var_D.parameters()), lr=learning_rate, weight_decay=weight_decay)
    scheduler = scheduler_case(scheduler_type, optimizer, learning_rate, n_iter)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_param("num_iter", n_iter)
        mlflow.log_param("initial_learning_rate", learning_rate)
        mlflow.log_param("schedule", scheduler_type)

        inputs, targets = train_data  # Extract train data
        dataset = IndexedDataset(inputs, targets)
        train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

        for iter in range(n_iter):
            model.train(True)
            for i, train_data in enumerate(train_dataloader):
                optimizer.zero_grad()  # Zero gradient for every batch
                loss = torch.zeros(1, device=device)
                inputs, targets, indices = train_data  # Extract train data
                mu, var = model(inputs)
                loss += loss_fn(mu, var, targets, indices = indices, hyperparameters=hyperparameters ,case=case, tau=tau, tau_adjustment=tau_adjustment)
                loss.backward()  # Compute the gradients and retain the graph
                optimizer.step()  # Take a step in the optimisation

            mlflow.log_metric("train_loss_per_epoch", loss.item() ,step=iter)
            print('Both Update - Epoch: %d - Loss: %f' % (iter, loss.item()))

            if iter % valid_freq == 0:
                model.eval()
                with torch.no_grad():
                    valid_inputs, valid_targets = valid_data  # Extract train data
                    mu_valid, var_valid = model(valid_inputs)
                    valid_loss_eval = loss_mse(mu_valid, valid_targets)
                    print('Epoch: %d MSE Valid: %f' % (iter, valid_loss_eval.item()))
                mlflow.log_metric("valid_eval", valid_loss_eval.item(),step=iter)  # Print Validation loss after epoch.

            if scheduler_type is not None:
                mlflow.log_metric("current_learning_rate_both", optimizer.param_groups[0]['lr'], step=iter)
                scheduler.step()

            if iter % save_freq == 0:
                mlflow.pytorch.log_model(model, '%s_epoch_%f' % (model_save_name, iter))

        mlflow.pytorch.log_model(model, model_save_name)
    return run_id