import os
import pickle
import numpy as np
import argparse
import sys
import logging
from collections import OrderedDict

import torch.nn as nn

from sklearn.linear_model import LogisticRegression

from logs.utils import get_chk_name
from contrastive_learning.contrastive_core import ContrastiveBase
from dl_training.training import BaseTrainer
from dl_training.testing import BaseTester
from architectures.alexnet import alexnet
from architectures.resnet import resnet18
from architectures.densenet import densenet121
from logs.utils import setup_logging

logger = logging.getLogger()

class RegressionTester(BaseTester):

    def __init__(self, args):
        super().__init__(args)
        self.loss = None # Useless here
        logger.debug(f"Metrics: {self.metrics}")
        logger.debug(f"Datamanager: {self.manager.__class__.__name__}")
        
        try:
            logger.debug("Net:")
            for k, v in self.net._modules.items():
                logger.debug(f"* {k}: {v.__class__.__name__}")
        except BaseException as e:
            logger.error(str(e))

    def build_network(self, **kwargs):
        _ = kwargs.pop("num_classes", 1)
        if self.args.net == "resnet18":
            n_embedding = kwargs.pop("n_embedding", 512)
            net = resnet18(n_embedding=n_embedding, **kwargs)
        elif self.args.net == "densenet121":
            n_embedding = kwargs.pop("n_embedding", 512)
            net = densenet121(n_embedding=n_embedding, **kwargs)
        elif self.args.net == "alexnet":
            n_embedding = kwargs.pop("n_embedding", 128) 
            net = alexnet(n_embedding=n_embedding, **kwargs)
        else:
            raise ValueError(f'Unknown network {self.args.net}')
        
        return nn.Sequential(OrderedDict([("encoder", net)]))
    
    def run(self):
        epochs_tested = self.get_epochs_to_test()
        runs_to_test = self.get_runs_to_test()
        for run in runs_to_test:
            for epoch in epochs_tested[run]:
                pretrained_path = os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, run, epoch))
                logger.debug(f"Pretrained path ({run}/{epoch}): {pretrained_path}")
                if self.args.outfile_name is None:
                    self.args.outfile_name = f"RegressionTester"
                exp_name = f"{self.args.outfile_name}_exp-{self.args.exp_name}_run-{run}_ep-{epoch}.pkl"
                model = ContrastiveBase(model=self.net,
                                        loss=self.loss,
                                        metrics=self.metrics,
                                        pretrained=pretrained_path,
                                        use_cuda=self.args.cuda)

                #class_weights = {"scz": 1.131, "asd": 1.584, "bd": 1.584}
                clf = LogisticRegression(penalty='l2', C=1.0, tol=0.0001, fit_intercept=True,
                                         class_weight="balanced",
                                         random_state=None, solver='lbfgs', max_iter=1000, verbose=self.args.verbose,
                                         n_jobs=self.args.num_cpu_workers)
                predictions = {}
                # Training
                logger.info(f"Fit logistic regression on training set")
                loader = self.manager.get_dataloader(train=True,
                                                     validation=True,
                                                     test=True,
                                                     run=run)
                z, labels = model.get_embeddings(loader.train)
                clf = clf.fit(z, labels)
                logger.debug(f"LogisticRegression classes: {clf.classes_}")
                for split in ("train", "validation", "test", "test_intra"): 
                    logger.info(f"Set: {split}")
                    if split == "test_intra":
                        loader = self.manager.get_dataloader(test_intra=True,
                                                             run=run)
                        z, labels = model.get_embeddings(loader.test)
                    else:
                        z, labels = model.get_embeddings(getattr(loader, split))
                    y_pred = clf.predict_proba(z)
                    if "test" in split:
                        split = {"test": "external test", "test_intra": "internal test"}[split]
                    predictions[split] = {"y_pred": y_pred,
                                          "y_true": labels,
                                          "metrics": {}}
                    for name, metric in model.metrics.items():
                        predictions[split]["metrics"][name] = metric(y_pred=y_pred, y_true=labels)
                
                with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                    pickle.dump(predictions, f)


def main(argv):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Train a linear classifier with embeddings')
    parser.add_argument("--root", type=str, required=True, help="Path to data root directory")
    parser.add_argument("--preproc", type=str, default='vbm', choices=['vbm', 'quasi_raw', "skeleton"])
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--outfile_name", type=str,
                        help="The output file name used to save the results in testing mode.")
    parser.add_argument("--pb", type=str, choices=["scz", "bd", "asd"])
    parser.add_argument("--run", nargs='+', type=int, help="Run indexes to test")
    parser.add_argument("--nb_runs", type=int, default=5)

    parser.add_argument("--net", type=str, help="Network to use")
    parser.add_argument("--model", type=str, help="Model to use", choices=["base", "SupCon"],
                        default="base")
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("--nb_epochs", type=int, default=300)

    parser.add_argument("--num_cpu_workers", type=int, default=3,
                        help="Number of workers assigned to do the preprocessing step "
                             "(used by DataLoader of Pytorch)")
    parser.add_argument("--sampler", choices=["random", "weighted_random", "sequential"], required=True)
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--cuda", type=bool, default=True, help="If True, executes the code on GPU")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate verbosity mode")

    args = parser.parse_args(argv)
    args.data_augmentation = None
    args.lr = 1e-4

    # Setup Logging
    setup_logging(level="debug" if args.verbose else "info",
                  logfile=os.path.join(args.checkpoint_dir, f"{args.exp_name}.log"))
    logger.info(f"Checkpoint directory : {args.checkpoint_dir}")

    tester = RegressionTester(args)
    tester.run()


if __name__ == "__main__":
    main(argv=sys.argv[1:])
