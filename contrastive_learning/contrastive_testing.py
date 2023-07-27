import os
import pickle
import argparse
import sys
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from logs.utils import get_chk_name
from dl_training.core import Base
from dl_training.training import BaseTrainer
from dl_training.testing import BaseTester
from logs.utils import setup_logging

logger = logging.getLogger()

# FIXME : metrics --> be compliant with array
# FIXME : improve it (metrics in particular) for age regression


class RegressionTester(BaseTester):

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.model, out_block="features", in_channels=1)
        self.manager = BaseTrainer.build_data_manager("base", args.pb, args.preproc, args.root, args.N_train_max,
                                                      sampler=args.sampler, batch_size=args.batch_size,
                                                      number_of_folds=args.nb_folds,
                                                      device=('cuda' if args.cuda else 'cpu'),
                                                      num_workers=args.num_cpu_workers,
                                                      pin_memory=True)
        self.loss = BaseTrainer.build_loss(args.model, args.pb, args.cuda, sigma=args.sigma)
        self.metrics = BaseTrainer.build_metrics(args.pb, args.model)
        if self.args.pretrained_path and self.manager.number_of_folds > 1:
            logger.warning('Several folds found while a unique pretrained path is set!')

    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                logger.debug(f"Pretrained path ({fold}/{epoch}): {pretrained_path}")
                if self.args.outfile_name is None:
                    self.args.outfile_name = f"RegressionTester_{self.args.exp_name}"
                exp_name = f"{self.args.outfile_name}_fold{fold}_epoch{epoch}.pkl"
                model = Base(model=self.net,
                             loss=self.loss,
                             metrics=self.metrics,
                             pretrained=pretrained_path,
                             load_optimizer=False,
                             use_cuda=self.args.cuda)

                class_weights = {"scz": 1.131, "asd": 1.584, "bipolar": 1.584, "sex": 1.0, "age": None}
                clf = LogisticRegression(penalty='l2', C=1.0, tol=0.0001, fit_intercept=True,
                                         class_weight=class_weights[self.args.pb],
                                         random_state=None, solver='lbfgs', max_iter=1000, verbose=0,
                                         n_jobs=self.args.num_cpu_workers)
                predictions = {}
                # Training
                logger.info(f"Train set")
                loader = self.manager.get_dataloader(train=True,
                                                     validation=True,
                                                     fold_index=fold)
                z, labels, _ = model.get_embeddings(loader.train)
                X = np.array(z)
                y_true = np.array(labels)
                clf = clf.fit(X, y_true)
                y_pred = clf.predict_proba(X)
                predictions["train"] = {"y_pred": y_pred,
                                        "y_true": labels}
                y_score = y_pred[:, 1]
                predictions["train"]["roc_auc"] = roc_auc_score(y_score=y_score,
                                                                y_true=y_true)
                predictions["train"]["balanced_accuracy"] = balanced_accuracy_score(y_pred=y_score > 0.5,
                                                                                    y_true=y_true)
                # Validation
                logger.info(f"Validation set")
                z, labels, _ = model.get_embeddings(loader.validation)
                X = np.array(z)
                y_true = np.array(labels)
                y_pred = clf.predict_proba(X)
                y_score = y_pred[:, 1]
                predictions["validation"] = {"y_pred": y_pred,
                                             "y_true": y_true}
                predictions["validation"]["roc_auc"] = roc_auc_score(y_score=y_score,
                                                                     y_true=y_true)
                predictions["validation"]["balanced_accuracy"] = balanced_accuracy_score(y_pred=y_score > 0.5,
                                                                                         y_true=y_true)
                # Tests
                for test in ["external", "internal"]:
                    logger.info(f"{test} test")
                    loader = self.manager.get_dataloader(test=(test == "external"),
                                                         test_intra=(test == "internal"),
                                                         fold_index=fold)
                    z, labels, _ = model.get_embeddings(loader.test, )
                    X = np.array(z)
                    y_true = np.array(labels)
                    y_pred = clf.predict_proba(X)
                    y_score = y_pred[:, 1]
                    predictions[f"{test} test"] = {"y_pred": y_pred,
                                                   "y_true": y_true}
                    predictions[f"{test} test"]["roc_auc"] = roc_auc_score(y_score=y_score,
                                                                           y_true=y_true)
                    predictions[f"{test} test"]["balanced_accuracy"] = balanced_accuracy_score(y_pred=y_score > 0.5,
                                                                                               y_true=y_true)
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
    parser.add_argument("--pb", type=str, choices=["age", "sex", "scz", "bipolar", "asd", "self_supervised"])
    parser.add_argument("--folds", nargs='+', type=int, help="Fold indexes to run during the training")
    parser.add_argument("--nb_folds", type=int, default=5)

    parser.add_argument("--net", type=str, help="Network to use")
    parser.add_argument("--model", type=str, help="Model to use", choices=["base", "SimCLR", "SupCon", "y-aware"],
                        default="base")
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("--nb_epochs", type=int, default=300)

    parser.add_argument("--num_cpu_workers", type=int, default=3,
                        help="Number of workers assigned to do the preprocessing step (used by DataLoader of Pytorch)")
    parser.add_argument("--sampler", choices=["random", "weighted_random", "sequential"], required=True)
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--sigma", type=float, help="Hyper-parameter for RBF kernel in self-supervised loss.", default=5)
    parser.add_argument("--cuda", type=bool, default=True, help="If True, executes the code on GPU")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate verbosity mode")

    args = parser.parse_args(argv)
    args.data_augmentation = None
    if args.pb in ["age", "sex"]:
        args.N_train_max = 1100
    else:
        args.N_train_max = None
    args.lr = 1e-4

    # Setup Logging
    setup_logging(level="debug" if args.verbose else "info",
                  logfile=os.path.join(args.checkpoint_dir, f"{args.exp_name}.log"))
    logger.info(f"Checkpoint directory : {args.checkpoint_dir}")

    tester = RegressionTester(args)
    tester.run()


if __name__ == "__main__":

    main(argv=sys.argv[1:])
