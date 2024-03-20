import os
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import wandb


def set_environment_variables(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = args.checkpoint_dir


def define_wandb_metrics(metrics):
    wandb.define_metric("epoch")
    for m in metrics.keys():
        wandb.define_metric(f"{m}/*", step_metric="epoch")


def main_log(train_history, valid_history):
    train_dict = train_history.to_dict()
    valid_dict = valid_history.to_dict()
    nb_trainings = len(train_dict["loss"])
    nb_epochs = len(train_dict["loss"][0])

    metrics = {
        "training_loss": train_dict["loss"],
        "validation_loss": valid_dict["validation_loss"]}
    if "roc_auc on validation set" in valid_dict.keys():
        metrics["roc_auc"] = valid_dict["roc_auc on validation set"]

    define_wandb_metrics(metrics)

    with_mean = {
        "training_loss": False,
        "validation_loss": True,
        "roc_auc": False
    }

    log_metrics(metrics, nb_trainings, nb_epochs, with_mean)

    plots = plot_training_curves(metrics, nb_trainings, nb_epochs, with_mean)
    wandb.log(plots)

    if "balanced_accuracy on validation set" in valid_dict.keys():
        metrics["balanced_accuracy"] = valid_dict['balanced_accuracy on validation set']
    update_summary(metrics, nb_trainings)


def plot_training_curves(metrics, nb_trainings, nb_epochs, with_mean=None):
    if with_mean is None:
        with_mean = {k: False for k in metrics.keys()}
    # Colors of the curves
    colors = [px.colors.qualitative.Plotly[f] for f in range(nb_trainings)]

    # Plots
    plots = {}
    x = [e for e in range(nb_epochs)]
    for k, v in metrics.items():
        plots[k] = go.Figure()
        for f in range(nb_trainings):
            plots[k].add_trace(go.Scatter(
                x=x, y=v[f],
                line_color=colors[f],
                name=str(f)))
        plots[k].update_layout(
            xaxis_title="Epoch",
            legend_title="N° of training :")

        if with_mean[k]:
            mean, std = np.mean(v, axis=0), np.std(v, axis=0)
            y, y_upper, y_lower = list(mean), list(mean + std), list(mean - std)
            x_rev, y_lower = x[::-1], y_lower[::-1]
            plots[f"{k}/mean"] = go.Figure()
            plots[f"{k}/mean"].add_trace(go.Scatter(
                x=x + x_rev,
                y=y_upper + y_lower,
                fill='toself',
                fillcolor='rgba(0,176,246,0.2)',
                line_color='rgba(255,255,255,0)',
                name=f"std {k}",
                showlegend=False,
            ))

            plots[f"{k}/mean"].add_trace(go.Scatter(
                x=x, y=y,
                line_color='rgb(0,176,246)',
                name=f"mean {k}",
            ))

            plots[f"{k}/mean"].update_layout(
                xaxis_title="Epoch")
    return plots


def log_metrics(metrics, nb_trainings, nb_epochs, with_mean=None):

    if with_mean is None:
        with_mean = {k: False for k in metrics.keys()}
    for epoch in range(nb_epochs):
        metric_logs = {"epoch": epoch}
        for k, v in metrics.items():
            for training in range(nb_trainings):
                metric_logs[f"{k}/training{training}"] = v[training][epoch]
            if with_mean[k]:
                metric_logs[f"{k}/mean"] = np.mean([v[f][epoch] for f in range(nb_trainings)])
        wandb.log(metric_logs)


def update_summary(metrics, nb_trainings, num_epoch=-1):

    for k, v in metrics.items():
        wandb.run.summary[f"{k}/mean"] = np.mean([v[f][num_epoch] for f in range(nb_trainings)])
        wandb.run.summary[f"{k}/std"] = np.std([v[f][num_epoch] for f in range(nb_trainings)])
