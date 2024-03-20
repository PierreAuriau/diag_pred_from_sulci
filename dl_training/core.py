# -*- coding: utf-8 -*-
"""
Core classes.
"""

import os
import pickle
from copy import deepcopy
import subprocess
from tqdm import tqdm
import numpy as np
import logging
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Package import
from logs.utils import get_chk_name
from logs.history import History
import dl_training.metrics as mmetrics

logger = logging.getLogger()

class Base(object):
    """ Class to perform classification.
    """
    def __init__(self, optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, use_cuda=False,
                 pretrained=None, use_multi_gpu=True, **kwargs):
        """ Class instantiation.

        Observers will be notified, allowed signals are:
        - 'before_epoch'
        - 'after_epoch'

        Parameters
        ----------
        optimizer_name: str, default 'Adam'
            the name of the optimizer: see 'torch.optim' for a description
            of available optimizer.
        learning_rate: float, default 1e-3
            the optimizer learning rate.
        loss_name: str, default 'NLLLoss'
            the name of the loss: see 'torch.nn' for a description
            of available loss.
        metrics: list of str
            a list of extra metrics that will be computed.
        use_cuda: bool, default False
            whether to use GPU or CPU.
        pretrained: path, default None
            path to the pretrained model or weights.
        use_multi_gpu: boolean, default True
            if several GPUs are available, use them during forward/backward pass
        kwargs: dict
            specify directly a custom 'model', 'optimizer' or 'loss'.
        """
        
        # Model
        self.model = kwargs.pop("model")
        if pretrained is not None:
            self.load_checkpoint(pretrained)
        
        # Loss
        self.loss = kwargs.pop("loss", None)
        if self.loss is None:
            if loss_name not in dir(torch.nn):
                raise ValueError("Loss '{0}' uknown: check available loss in "
                                 "'pytorch.nn'.")
            self.loss = getattr(torch.nn, loss_name)()
        
        # Optimizer
        self.optimizer = kwargs.pop("optimizer", None)
        if self.optimizer is None:
            if optimizer_name in dir(torch.optim):
                self.optimizer = getattr(torch.optim, optimizer_name)(
                    self.model.parameters(),
                    lr=learning_rate)
            else:
                raise ValueError("Optimizer '{0}' uknown: check available optimizer in 'pytorch.optim'.")

        # Metrics
        self.metrics = {}
        for name in (metrics or []):
            if name not in mmetrics.METRICS:
                raise ValueError("Metric '{0}' not yet supported: you can try "
                                 "to fill the 'METRICS' factory, or ask for "
                                 "some help!".format(name))
            self.metrics[name] = mmetrics.METRICS[name]
        # Device
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: unset 'use_cuda' parameter.")
        if use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)

        self.model = self.model.to(self.device)
        
    def training(self, manager, nb_epochs: int, checkpointdir=None,
                 runs=None, scheduler=None, with_validation=True,
                 nb_epochs_per_saving=1, exp_name=None, **kwargs_train):
        """ Train the model.

        Parameters
        ----------
        manager: a dl_training DataManager
            a manager containing the train and validation data.
        nb_epochs: int, default 100
            the number of epochs.
        checkpointdir: str, default None
            a destination traininger where intermediate architectures/historues will be
            saved.
        run: int or [int] default None
            the index(es) of the run(s) to use for the training, default use all the
            available runs.
        scheduler: torch.optim.lr_scheduler, default None
            a scheduler used to reduce the learning rate.
        with_validation: bool, default True
            if set use the validation dataset.
        nb_epochs_per_saving: int, default 1,
            the number of epochs after which the model+optimizer's parameters are saved
        exp_name: str, default None
            the experience name that will be launched
        Returns
        -------
        train_history, valid_history: History
            the train/validation history.
        """
        if not os.path.exists(checkpointdir):
            raise FileNotFoundError(f"Checkpoint directory does not exists ! {checkpointdir}")
        train_history = History(name="Train")
        if with_validation is not None:
            valid_history = History(name="Validation")
        else:
            valid_history = None
        logger.info(f"Loss : {self.loss}")
        logger.info(f"Optimizer : {self.optimizer}")
        if isinstance(runs, int):
            runs = [runs]
        elif isinstance(runs, list):
            runs = runs
        else:
            runs = range(manager.get_number_of_runs())
        init_optim_state = deepcopy(self.optimizer.state_dict())
        init_model_state = deepcopy(self.model.state_dict())
        scaler = kwargs_train.get("gradscaler")
        if scheduler is not None:
            init_scheduler_state = deepcopy(scheduler.state_dict())
        for run in runs:
            # Initialize everything before optimizing on a new training
            if scaler is not None:
                kwargs_train["gradscaler"] = GradScaler()
                logger.info("GradScaler activated")
            self.optimizer.load_state_dict(init_optim_state)
            self.model.load_state_dict(init_model_state)
            if scheduler is not None:
                scheduler.load_state_dict(init_scheduler_state)
            loader = manager.get_dataloader(
                train=True,
                validation=True,
                run=run)
            min_loss, best_model, best_epoch = None, None, None
            for epoch in range(nb_epochs):
                loss, values = self.train(loader.train, run, epoch, **kwargs_train)

                train_history.log((run, epoch), loss=loss, **values)
                train_history.summary()
                if scheduler is not None:
                    scheduler.step()
                    logger.info('Scheduler lr: {}'.format(scheduler.get_last_lr()))
                    logger.info('Optimizer lr: %f' % self.optimizer.param_groups[0]['lr'])
                if (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) \
                        and epoch > 0:
                    self.save_checkpoint(
                        model=self.model.state_dict(),
                        epoch=epoch,
                        run=run,
                        outdir=checkpointdir,
                        name=exp_name,
                        optimizer=self.optimizer.state_dict())
                    train_history.save(
                        outdir=checkpointdir,
                        exp_name=exp_name,
                        epoch=epoch,
                        run=run,
                        to_dict=True)
                if with_validation:
                    y_pred, y_true, loss, values = self.test(loader.validation, **kwargs_train)
                    valid_history.log((run, epoch), validation_loss=loss, y_pred=y_pred, y_true=y_true, **values)
                    valid_history.summary()
                    if (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) \
                            and epoch > 0:
                        valid_history.save(
                            outdir=checkpointdir,
                            exp_name=exp_name,
                            epoch=epoch,
                            run=run,
                            to_dict=True)
                if min_loss is None or loss < min_loss:
                    min_loss = loss
                    best_epoch = epoch
                    best_model = deepcopy(self.model.state_dict())
            if best_epoch % nb_epochs_per_saving != 0:
                self.save_checkpoint(
                    model=best_model,
                    epoch=best_epoch,
                    run=run,
                    outdir=checkpointdir,
                    name=exp_name)
        return train_history, valid_history

    def train(self, loader, run=None, epoch=None, **kwargs):
        """ Train the model on the trained data.

        Parameters
        ----------
        loader: a pytorch Dataloader
        run: number of the run
        epoch: number of the epoch

        Returns
        -------
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """
        scaler = kwargs.get("gradscaler")
        self.model.train()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc=f"Mini-Batch ({run},{epoch})")

        values = {}
        losses = []
        y_pred = []
        y_true = []
        for dataitem in loader:
            pbar.update()
            inputs = dataitem.inputs
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            list_targets = []
            _targets = []
            for item in (dataitem.outputs, dataitem.labels):
                if item is not None:
                    _targets.append(item.to(self.device))
            if len(_targets) == 1:
                _targets = _targets[0]
            list_targets.append(_targets)

            self.optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    batch_loss = self.loss(outputs, *list_targets)
                scaler.scale(batch_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(inputs)
                batch_loss = self.loss(outputs, *list_targets)
                batch_loss.backward()
                self.optimizer.step()

            losses.append(float(batch_loss))
            y_pred.extend(outputs.detach().cpu().numpy())
            y_true.extend(list_targets[0].detach().cpu().numpy())

            aux_losses = (self.model.get_aux_losses() if hasattr(self.model, 'get_aux_losses') else dict())
            aux_losses.update(self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())

            for name, aux_loss in aux_losses.items():
                if name not in values:
                    values[name] = 0
                values[name] += float(aux_loss) / nb_batch
        loss = np.mean(losses)
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        for name, metric in self.metrics.items():
            values[name] = float(metric(y_pred, y_true))
        #values["y_pred"] = y_pred
        #values["y_true"] = y_true
        pbar.close()
        return loss, values

    def testing(self, loader: DataLoader, saving_dir=None, exp_name=None, **kwargs):
        """ Evaluate the model.

        Parameters
        ----------
        loader: a pytorch DataLoader
        saving_dir: str path to the saving directory
        exp_name: str, name of the experiments that is used to derive the output file name of testing results.
        Returns
        -------
        y: array-like
            the predicted data.
        X: array-like
            the input data.
        y_true: array-like
            the true data if available.
        loss: float
            the value of the loss function if true data availble.
        values: dict
            the values of the metrics if true data availble.
        """
        set_name = "internal test" if exp_name.startswith("internal") else "external test"
        y_pred, y_true, loss, values = self.test(loader, set_name, **kwargs)

        if saving_dir is not None:
            if not os.path.isdir(saving_dir):
                subprocess.check_call(['mkdir', '-p', saving_dir])
                logger.info("Directory %s created." % saving_dir)
            with open(os.path.join(saving_dir, f"Test-{exp_name}.pkl"), 'wb') as f:
                pickle.dump({'y_pred': y_pred, 'y_true': y_true, 'loss': loss, 'metrics': values}, f)

        return y_pred, y_true, loss, values

    def test(self, loader, set_name="validation", **kwargs):
        """ Evaluate the model on the tests or validation data.

        Parameter
        ---------
        loader: a pytorch Dataset
            the data loader.
        Returns
        -------
        y: array-like
            the predicted data.
        y_true: array-like
            the true data
        X: array_like
            the input data
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """

        self.model.eval()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")
        loss = 0
        values = {}

        with torch.no_grad():
            y_pred, y_true = [], []

            for dataitem in loader:
                pbar.update()
                inputs = dataitem.inputs
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                list_targets = []
                targets = []
                for item in (dataitem.outputs, dataitem.labels):
                    if item is not None:
                        targets.append(item.to(self.device))
                        y_true.extend(item.cpu().detach().numpy())
                if len(targets) == 1:
                    targets = targets[0]
                elif len(targets) == 0:
                    targets = None
                if targets is not None:
                    list_targets.append(targets)

                outputs = self.model(inputs)

                if len(list_targets) > 0:
                    batch_loss = self.loss(outputs, *list_targets)
                    loss += float(batch_loss) / nb_batch

                y_pred.extend(outputs.cpu().detach().numpy())                   

                aux_losses = (self.model.get_aux_losses() if hasattr(self.model, 'get_aux_losses') else dict())
                aux_losses.update(self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())
                for name, aux_loss in aux_losses.items():
                    name += f" on {set_name} set"
                    if name not in values:
                        values[name] = 0
                    values[name] += aux_loss / nb_batch
            y_pred = np.asarray(y_pred)
            y_true = np.asarray(y_true)
            
            # Now computes the metrics with (y, y_true)
            for name, metric in self.metrics.items():
                name += f" on {set_name} set"
                values[name] = metric(y_pred, y_true)
        pbar.close()
        return y_pred, y_true, loss, values

    def get_embeddings(self, loader):
        """ Get the outputs of the model.

        Parameter
        ---------
        loader: a pytorch Dataset
            the data loader.
        Returns
        -------
        z: array-like
            the embeddings
        labels: array-like
            the true data
        X: array_like
            the input data
        """

        self.model.eval()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")

        with torch.no_grad():
            z, labels, X = [], [], []

            for dataitem in loader:
                pbar.update()
                inputs = dataitem.inputs
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                for item in (dataitem.outputs, dataitem.labels):
                    if item is not None:
                        labels.extend(item.cpu().detach().numpy())
                outputs = self.model(inputs)
                z.extend(outputs.cpu().detach().numpy())

                if isinstance(inputs, torch.Tensor):
                    X.extend(inputs.cpu().detach().numpy())

            pbar.close()

        return z, labels, X
    
    def load_checkpoint(self, pretrained, strict=False):
            checkpoint = None
            try:
                checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
                logger.debug(f"Checkpoint Load : {pretrained}")
            except BaseException as e:
                logger.error('Impossible to load the checkpoint: %s' % str(e))
            if checkpoint is not None:
                if hasattr(checkpoint, "state_dict"):
                    self.model.load_state_dict(checkpoint.state_dict())
                    logger.debug(f"State Dict Loaded")
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        try:
                            for key in list(checkpoint['model'].keys()):
                                if key.replace('module.', '') != key:
                                    checkpoint['model'][key.replace('module.', '')] = checkpoint['model'][key]
                                    del(checkpoint['model'][key])
                            unexpected = self.model.load_state_dict(checkpoint["model"], strict=strict)
                            logger.info('Model loading info: {}'.format(unexpected))
                            logger.info('Model loaded')
                        except BaseException as e:
                            logger.error('Error while loading the model\'s weights: %s' % str(e))
                            raise ValueError("")
                else:
                    self.model.load_state_dict(checkpoint)
    
    def save_checkpoint(self, epoch, run, outdir, name=None, **kwargs):
        name = get_chk_name(name, run, epoch)
        outfile = os.path.join(outdir, name)
        if "model" not in kwargs.keys():
            kwargs["model"] = self.model.state_dict()
        if "optimizer" not in kwargs.keys():
            kwargs["optimizer"] = self.optimizer.state_dict()
        torch.save(
            {"name": name,
             "run": run,
             "epoch": epoch,
             **kwargs}, outfile)
