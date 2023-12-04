# -*- coding: utf-8 -*-

import os
import pickle
import nibabel
from logs.utils import get_chk_name
from dl_training.core import Base
from contrastive_learning.contrastive_core import ContrastiveBase
from dl_training.training import BaseTrainer

logger = logging.getLogger()


class BaseTester:

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.model, num_classes=1, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args.model, args.pb, args.preproc, args.root,
                                                      sampler=args.sampler, batch_size=args.batch_size,
                                                      number_of_folds=args.nb_folds,
                                                      data_augmentation=args.data_augmentation,
                                                      device=('cuda' if args.cuda else 'cpu'),
                                                      num_workers=args.num_cpu_workers,
                                                      pin_memory=True)
        self.loss = BaseTrainer.build_loss(args.model, args.pb, args.cuda)
        self.metrics = BaseTrainer.build_metrics(args.pb, args.model)
        self.kwargs_test = dict()

        if self.args.pretrained_path and self.manager.number_of_folds > 1:
            logger.warning('Several folds found while a unique pretrained path is set!')

    def get_folds_to_test(self):
        if self.args.folds is not None and len(self.args.folds) > 0:
            folds = self.args.folds
        else:
            folds = list(range(self.args.nb_folds))
        return folds

    def get_epochs_to_test(self):
        # Get the last point and tests it, for each fold
        epochs_tested = [[self.args.nb_epochs - 1] for _ in range(self.args.nb_folds)]

        return epochs_tested

    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            tests = ["", "Intra_"]
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                logger.debug(f"Pretrained path : {pretrained_path}")
                for t in tests:
                    if self.args.outfile_name is None:
                        outfile = "%s%s" % (t, self.args.exp_name)
                    else:
                        outfile = "%s%s" % (t, self.args.outfile_name)
                    exp_name = outfile + "_fold{}_epoch{}".format(fold, epoch)
                    loader = self.manager.get_dataloader(test=(t == ""),
                                                         test_intra=(t != ""),
                                                         fold_index=fold)
                    model = Base(model=self.net, loss=self.loss,
                                 metrics=self.metrics,
                                 pretrained=pretrained_path,
                                 use_cuda=self.args.cuda)
                    model.testing(loader.test, saving_dir=self.args.checkpoint_dir, exp_name=exp_name,
                                  **self.kwargs_test)
