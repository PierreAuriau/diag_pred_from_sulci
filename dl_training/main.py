import argparse
import torch
import logging
import wandb
import os
from dl_training.training import BaseTrainer
from dl_training.testing import OpenBHBTester
from logs.utils import save_hyperparameters, setup_logging


if __name__ == "__main__":

    logger = logging.getLogger()

    parser = argparse.ArgumentParser()

    # Data location + saving paths
    parser.add_argument("--root", type=str, required=True, help="Path to data root directory")
    parser.add_argument("--preproc", type=str, default='vbm', choices=['vbm', 'quasi_raw', "skeleton"])
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--outfile_name", type=str,
                        help="The output file name used to save the results in testing mode.")
    parser.add_argument("--N_train_max", type=int, default=None, help="Maximum number of training samples "
                                                                      "to be used per fold")
    parser.add_argument("--pb", type=str, choices=["age", "sex", "scz", "bipolar", "asd", "self_supervised"])
    parser.add_argument("--folds", nargs='+', type=int, help="Fold indexes to run during the training")
    parser.add_argument("--nb_folds", type=int, default=5)

    # Important: what model do we use
    parser.add_argument("--net", type=str, help="Network to use")
    parser.add_argument("--model", type=str, help="Model to use", choices=["base", "SimCLR", "SupCon", "y-aware"],
                        default="base")

    # Depends on available CPU/GPU memory
    parser.add_argument("-b", "--batch_size", type=int, required=True)

    parser.add_argument("--nb_epochs_per_saving", type=int, default=5)
    parser.add_argument("--manual_seed", type=int, help="The manual seed to give to pytorch.")

    # Optimizer hyper-parameters
    parser.add_argument("--lr", type=float, required=True, help="Initial learning rate")
    parser.add_argument("--gamma_scheduler", type=float, required=True)
    parser.add_argument("--nb_epochs", type=int, default=300)
    parser.add_argument("--step_size_scheduler", nargs="+", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=5e-5)

    # Dataloader: set them
    parser.add_argument("--num_cpu_workers", type=int, default=3,
                        help="Number of workers assigned to do the preprocessing step (used by DataLoader of Pytorch)")
    parser.add_argument("--sampler", choices=["random", "weighted_random", "sequential"], required=True)
    parser.add_argument("--data_augmentation", type=str, nargs="+", default=None,
                        help="Data Augmentation for contrastive models")

    # Self-supervised learning
    parser.add_argument("--temperature", type=float, help="Hyper-parameter for contrastive loss", default=0.1)
    parser.add_argument("--sigma", type=float, help="Hyper-parameter for RBF kernel in self-supervised loss.", default=5)

    # Transfer Learning
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--load_optimizer", action="store_true", help="If <pretrained_path> is set, loads also the "
                                                                      "optimizer's weigth")

    # This code can be executed on CPU or GPU
    parser.add_argument("--cuda", type=bool, default=True, help="If True, executes the code on GPU")

    # Kind of tests
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # Wandb implementation for sweeps
    parser.add_argument("--sweep", action="store_true")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate verbosity mode")

    args = parser.parse_args()

    # ---------------------------------------------------------------------------------------------------------------- #

    # Create saving directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Setup Logging
    setup_logging(level="debug" if args.verbose else "info",
                  logfile=os.path.join(args.checkpoint_dir, f"{args.exp_name}.log"))

    if args.sweep:
        # Wandb
        from logs.wandb_log import set_environment_variables
        set_environment_variables(args=args)
        run = wandb.init()
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, run.name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        setup_logging(level="debug" if args.verbose else "info",
                      logfile=os.path.join(args.checkpoint_dir, f"{args.exp_name}.log"))

    logger.info(f"Checkpoint directory : {args.checkpoint_dir}")

    if not torch.cuda.is_available():
        args.cuda = False
        logger.warning("cuda is not available and has been disabled.")

    if args.manual_seed:
        torch.manual_seed(args.manual_seed)

    if not args.train and not args.test:
        args.train = True
        logger.info("No mode specify: training mode is set automatically")

    if args.train:
        save_hyperparameters(args)
        logger.info("Hyperparameters saved")
        trainer = BaseTrainer(args)
        train_history, valid_history = trainer.run()
        if args.sweep:
            from logs.wandb_log import main_log
            main_log(train_history, valid_history)
        # do not consider the pretrained path anymore since it will be eventually computed automatically
        args.pretrained_path = None

    if args.test:
        tester = OpenBHBTester(args)
        tester.run()





