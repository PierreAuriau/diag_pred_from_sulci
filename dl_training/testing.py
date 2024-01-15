# -*- coding: utf-8 -*-

import os
import pickle
import logging
from logs.utils import get_chk_name
from dl_training.core import Base
from contrastive_learning.contrastive_core import ContrastiveBase
from dl_training.training import BaseTrainer

logger = logging.getLogger()


class BaseTester:

    def __init__(self, args):
        self.args = args
        self.net = self.build_network(num_classes=1, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(model=args.model, pb=args.pb, preproc=args.preproc, root=args.root,
                                                      sampler=args.sampler, batch_size=args.batch_size,
                                                      nb_runs=args.nb_runs,
                                                      data_augmentation=args.data_augmentation,
                                                      device=('cuda' if args.cuda else 'cpu'),
                                                      num_workers=args.num_cpu_workers,
                                                      pin_memory=True)
        self.loss = BaseTrainer.build_loss(args.model, args.pb, args.cuda)
        self.metrics = self.build_metrics()

    def get_runs_to_test(self):
        if self.args.runs is not None and len(self.args.runs) > 0:
            runs = self.args.runs
        else:
            runs = list(range(self.args.nb_runs))
        return runs

    def get_epochs_to_test(self):
        # Get the last point and tests it, for each training
        epochs_tested = [[self.args.nb_epochs - 1] for _ in range(self.args.nb_runs)]
        return epochs_tested
    
    def build_network(self, **kwargs):
        return BaseTrainer.build_network(name=self.args.net, model=self.args.model, **kwargs)
    
    def build_metrics(self):
        return BaseTrainer.build_metrics(self.args.model)

    def run(self):
        epochs_tested = self.get_epochs_to_test()
        runs_to_test = self.get_runs_to_test()
        for run in runs_to_test:
            tests = ["internal", "external"]
            for epoch in epochs_tested[run]:
                pretrained_path = os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, run, epoch))
                logger.debug(f"Pretrained path : {pretrained_path}")
                for t in tests:
                    if self.args.outfile_name is None:
                        outfile = f"{t}_exp-{self.args.exp_name}"
                    else:
                        outfile = f"{t}_{self.args.outfile_name}"
                    exp_name = outfile + f"_run-{run}_ep-{epoch}"
                    loader = self.manager.get_dataloader(test=(t == "external"),
                                                         test_intra=(t == "internal"),
                                                         run=run)
                    model = Base(model=self.net, loss=self.loss,
                                 metrics=self.metrics,
                                 pretrained=pretrained_path,
                                 use_cuda=self.args.cuda)
                    model.testing(loader.test, saving_dir=self.args.checkpoint_dir, exp_name=exp_name)

