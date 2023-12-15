# -*- coding: utf-8 -*-

import logging
import os
import numpy as np
import torch
import torch.nn as nn

from dl_training.core import Base
from contrastive_learning.contrastive_core import ContrastiveBase
from dl_training.datamanager import ClinicalDataManager
from dl_training.losses import SupConLoss
from architectures.alexnet import AlexNet3D_Dropout
from architectures.resnet import resnet18
from architectures.densenet import densenet121

logger = logging.getLogger()


class BaseTrainer:

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.model, num_classes=1, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(model=args.model, pb=args.pb, 
                                                      preproc=args.preproc, root=args.root,
                                                      sampler=args.sampler, batch_size=args.batch_size,
                                                      number_of_trainings=args.nb_trainings,
                                                      data_augmentation=args.data_augmentation,
                                                      device=('cuda' if args.cuda else 'cpu'),
                                                      num_workers=args.num_cpu_workers,
                                                      pin_memory=True)
        self.loss = BaseTrainer.build_loss(args.model, args.pb, args.cuda, temperature=args.temperature)
        self.metrics = BaseTrainer.build_metrics(args.pb, args.model)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = self.build_scheduler(args.step_size_scheduler, args.gamma_scheduler)

        if args.model == "SupCon":
            model_cls = ContrastiveBase
        else:
            model_cls = Base

        self.model = model_cls(model=self.net,
                               metrics=self.metrics,
                               pretrained=args.pretrained_path,
                               use_cuda=args.cuda,
                               loss=self.loss,
                               optimizer=self.optimizer)
        # logger.debug(f"net : {self.net}")
        logger.debug(f"manager : {self.manager}")
        logger.debug(f"loss : {self.loss}")
        logger.debug(f"metrics : {self.metrics}")
        logger.debug(f"model_cls : {model_cls}")

    def run(self):
        with_validation = True
        kwargs_train = {}
        if self.args.cuda:
            kwargs_train["gradscaler"] = True
        train_history, valid_history = self.model.training(self.manager,
                                                           nb_epochs=self.args.nb_epochs,
                                                           scheduler=self.scheduler,
                                                           with_validation=with_validation,
                                                           checkpointdir=self.args.checkpoint_dir,
                                                           nb_epochs_per_saving=self.args.nb_epochs_per_saving,
                                                           exp_name=self.args.exp_name,
                                                           training_index=self.args.training_index,
                                                           **kwargs_train)

        return train_history, valid_history

    def build_scheduler(self, step_size, gamma):
        if len(step_size) == 1:
            return torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=gamma,
                                                   step_size=step_size[0])
        elif len(step_size) > 1:
            num_epochs = list(np.cumsum(step_size))
            if num_epochs[-1] > self.args.nb_epochs:
                raise ValueError(f"Step sizes of the scheduler exceed the number of epochs "
                                 f"({num_epochs[-1]}/{self.args.nb_epochs})")
            while num_epochs[-1] + step_size[-1] < self.args.nb_epochs:
                num_epochs.append(num_epochs[-1] + step_size[-1])
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma,
                                                        milestones=num_epochs)
        else:
            raise NotImplementedError(f"Wrong step size scheduler : {step_size}")

    @staticmethod
    def build_metrics(pb, model):
        if model == "SupCon":
            # FIXME : metrics for SupCon
            metrics = ["accuracy"]
        else:
            metrics = ["balanced_accuracy", "roc_auc"]
        logger.debug(f"Metrics : {metrics}")
        return metrics

    @staticmethod
    def build_loss(model, pb, cuda, **kwargs):
        if model == "SupCon":
            temperature = kwargs.get("temperature", 0.1)
            loss = SupConLoss(temperature=temperature, base_temperature=temperature)
        else:
            # Binary classification tasks
            # Balanced BCE loss
            pos_weights = {"scz": 1.131, "asd": 1.584, "bd": 1.584, "sex": 1.0}
            loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights[pb], dtype=torch.float32,
                                                                device=('cuda' if cuda else 'cpu')))
        return loss

    @staticmethod
    def build_network(name, model, **kwargs):
        # one output for BCE loss and L1 loss. Last layers removed for self-supervision
        num_classes = kwargs.pop("num_classes", 1)
        out_block = kwargs.pop("out_block", None)
        logger.debug(f"Out block of net : {out_block}")
        logger.debug(f"Num classes for net : {num_classes}")
        if out_block is None and model in ["SimCLR", "SupCon", "y-aware"]:
            out_block = "simCLR"
            logger.info(f"For contrastive learning, the outblock of the net is set to : '{out_block}'")
        if name == "resnet18":
            net = resnet18(num_classes=num_classes, out_block=out_block, **kwargs)
        elif name == "densenet121":
            net = densenet121(num_classes=num_classes, out_block=out_block, **kwargs)
        elif name == "alexnet":  # AlexNet 3D version derived from Abrol et al., 2021
            if model == "SupCon":
                raise NotImplementedError("AlexNet not implemented for contrastive learning")
            else:
                net = AlexNet3D_Dropout(num_classes=num_classes)
        else:
            raise ValueError('Unknown network %s' % name)
        return net

    @staticmethod
    def build_data_manager(root, preproc, pb, model, **kwargs):
        kwargs["labels"] = ["diagnosis"]
        kwargs["db"] = pb
        kwargs["model"] = model
        logger.debug(f"Kwargs DataManager : {kwargs}")
        _manager_cls = ClinicalDataManager
        logger.debug(f"Datamanager : {_manager_cls}")
        manager = _manager_cls(root, preproc, **kwargs)
        return manager
    
    def __str__(self):
        return "BaseTrainer"

