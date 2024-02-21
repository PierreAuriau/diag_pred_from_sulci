import os
import sys
import argparse
import torch
import logging

from dl_training.training import BaseTrainer
from dl_training.testing import BaseTester
from logs.utils import save_hyperparameters, setup_logging

logger = logging.getLogger()

# UPDATEs :
# TODO : update save / load in contrastive core to save/load only encoder
# * how to save and load model ? 
# * Sequential class with get_embeddings param <https://discuss.pytorch.org/t/save-part-of-the-model/28519>
# * contrastive base --> save only encoder ?
# * contrastive testing --> load only encoder
# TODO : update contrastive_tester
# TODO : python script for creating pickles
# TODO : python script for saliency maps

def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Training and testing DL models.')

    # Data location + saving paths
    parser.add_argument("--root", type=str, required=True, 
                        help="Path to data root directory.")
    parser.add_argument("--preproc", type=str, default='no', choices=["no", "smoothing"])
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Folder where all the data will be saved.")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Filename used to save models.")
    parser.add_argument("--outfile_name", type=str,
                        help="The output file name used to save the results in testing mode.")
    parser.add_argument("--pb", type=str, choices=["scz", "bd", "asd"], required=True,
                        help="Diagnosis to predict.")

    parser.add_argument("--nb_runs", type=int, default=3,
                        help="Number of trainings with a different random initialization "
                             "of network weights. Default is: 3")
    parser.add_argument("--runs", nargs='+', type=int, 
                        help="Run indexes of models to train or test.")

    # Important: what model do we use
    parser.add_argument("--net", type=str, choices=["densenet121", "alexnet", "resnet18"], default="densenet121",
                        help="Architecture of network to use. Default is : densenet121")
    parser.add_argument("--model", type=str,  choices=["base", "SupCon"], default="base",
                        help="Model to use. Default is: base")

    # Depends on available CPU/GPU memory
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Number of images in the mini-batch. Default is: 32")
    parser.add_argument("--nb_epochs", type=int, default=50,
                        help="Number of epochs for training. Default is: 50.")
    parser.add_argument("--nb_epochs_per_saving", type=int, default=10,
                        help="Frequency at which model is saved during training. Default is: 10")

    # Optimizer hyper-parameters
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Initial learning rate. Default is: 1e-4")
    parser.add_argument("--gamma_scheduler", type=float, default=0.2,
                        help="Decreasing factor of the learning rate. Default is: 0.2")
    parser.add_argument("--step_size_scheduler", nargs="+", type=int, default=10,
                        help="Frequency at which the learning rate is decreased. "
                        "You can choose epochs at which the learning rate is decreased by passing several values. "
                        "Default is: 10")
    parser.add_argument("--weight_decay", type=float, default=5e-5,
                        help="Ponderation for the L2 penalty on the network's weights. Default is: 5e-5")

    # Dataloader: set them
    parser.add_argument("--num_cpu_workers", type=int, default=3,
                        help="Number of workers assigned to do the preprocessing step (used by DataLoader of Pytorch). "
                        "Default is: 3")
    parser.add_argument("--sampler", choices=["random", "weighted_random", "sequential"], default="random",
                        help="Image sampling for mini-batchs. Default is: random")

    # Self-supervised learning
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Hyper-parameter for SupCon loss. Default is: 0.1")
    parser.add_argument("--data_augmentation", type=str, nargs="+", default=None,
                    help="Data Augmentation for contrastive models.")
    
    # This code can be executed on CPU or GPU
    parser.add_argument("--cuda", type=bool, default=True, 
                        help="If True, executes the code on GPU. Default is: True.")

    # Kind of tests
    parser.add_argument("--train", action="store_true", help="If set, train model.")
    parser.add_argument("--test", action="store_true", help="If set, test model.")

    # Wandb implementation for sweeps
    parser.add_argument("--sweep", action="store_true", help="If set, launch wandb monitoring.")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate verbosity mode")

    args = parser.parse_args(argv)

    return args


def main(argv):
    
    args = parse_args(argv)
    
    # Create saving directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Setup Logging
    setup_logging(level="debug" if args.verbose else "info",
                  logfile=os.path.join(args.checkpoint_dir, f"exp-{args.exp_name}.log"))

    if args.sweep:
        # Wandb
        import wandb
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

    if args.test:
        tester = BaseTester(args)
        tester.run()


if __name__ == "__main__":
    main(sys.argv[1:])
