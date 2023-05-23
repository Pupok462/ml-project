import numpy as np
from tqdm.notebook import tqdm
import torch
import torchmetrics


def train_lstm(n_epochs, model, loss_fn, optimizer, train_loader, val_loader, device):
    train_losses = []
    val_losses = []
    metric_score = []
    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_losses_per_epoch = []
        for X_batch, y_real in train_loader:
            optimizer.zero_grad()

            forecasts = model(X_batch.to(device).float())

            loss = loss_fn(forecasts, y_real.to(device))
            loss.backward()
            optimizer.step()

            train_losses_per_epoch.append(loss.item())
        train_losses.append(np.mean(train_losses_per_epoch))

        model.eval()
        val_losses_per_epoch = []
        metric_score_per_epoch = []
        with torch.no_grad():
            for X_batch_val, y_real_val in val_loader:
                y_pred_val = model(X_batch_val.to(device).float())
                loss = loss_fn(y_pred_val, y_real_val.to(device).float())
                score = torchmetrics.functional.mean_absolute_percentage_error(y_pred_val,
                                                                               y_real_val.to(device).float())
                metric_score_per_epoch.append(score.cpu().detach().numpy())
                val_losses_per_epoch.append(loss.item())

        metric_score.append(np.mean(metric_score_per_epoch))
        val_losses.append(np.mean(val_losses_per_epoch))

    return train_losses, val_losses, metric_score


def save_model_for_inference(model):
    pass