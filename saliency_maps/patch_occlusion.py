# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import time

import torch
from torchvision.transforms.transforms import Compose

from logs.utils import setup_logging, get_chk_name
from dl_training.core import Base
from dl_training.training import BaseTrainer
from img_processing.augmentations import Cutout

logger = logging.getLogger()


class PatchOcclusion():

    def __init__(self, args):
        self.args = args
        self.net = self.build_network(num_classes=1, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(model="base", pb=args.pb, preproc=args.preproc, root=args.root,
                                                      sampler="sequential", batch_size=args.batch_size,
                                                      nb_runs=args.nb_runs,
                                                      data_augmentation=None,
                                                      device=('cuda' if args.cuda else 'cpu'),
                                                      num_workers=args.num_cpu_workers,
                                                      pin_memory=True)
        self.loss = BaseTrainer.build_loss(model="base", pb=args.pb, cuda=args.cuda)
        self.metrics = BaseTrainer.build_metrics(model="base")
        self.kwargs_test = dict()

        self.patch_size = args.patch_size
        self.step = args.step

        if self.args.pretrained_path and self.manager.number_of_runs > 1:
            logger.warning('Several runs found while a unique pretrained path is set!')

    def run(self):
        if self.args.outfile_name is None:
            self.args.outfile_name = f"PatchOcclusion_exp-{self.args.exp_name}"
        epochs_tested = self.get_epochs_to_test()
        runs_to_test = self.get_runs_to_test()
        for run in runs_to_test:
            for epoch in epochs_tested[run]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, run, epoch))
                logger.debug(f"Pretrained path : {pretrained_path}")

                pck_name = f"{self.args.outfile_name}_run-{run}_epoch-{epoch}.pkl"

                relevance_maps = {}
                for t in ["internal", "external"]:
                    t_name = {"internal": "test_intra", "external": "test"}[t]
                    loader = self.manager.get_dataloader(test=(t == "external"),
                                                         test_intra=(t == "internal"),
                                                         run_index=run)
                    model = Base(model=self.net, loss=self.loss,
                                 metrics=self.metrics,
                                 pretrained=pretrained_path,
                                 use_cuda=self.args.cuda)
                    
                    # Probabilities without occlusions
                    unoccluded_prob = []
                    model.model.eval()
                    pbar = tqdm(total=len(loader.test), desc="Mini-Batch")
                    with torch.no_grad():
                        for dataitem in loader.test:
                            pbar.update()
                            inputs = dataitem.inputs
                            if isinstance(inputs, torch.Tensor):
                                inputs = inputs.to(torch.device("cuda" if self.args.cuda 
                                                                else "cpu"))
                            outputs = model.model(inputs)
                            outputs = torch.sigmoid(outputs)
                            unoccluded_prob.extend(outputs.cpu().detach().numpy())
                    pbar.close()
                    unoccluded_prob = np.asarray(unoccluded_prob)
                    logger.debug(f"Unoccluded Probability : {unoccluded_prob.shape}")

                    # Probabilities with occlusions
                    # For occlusion, input_transform = Cutout
                    path2coord = os.path.join(self.args.root, f"coordinates_{self.args.pb}_patchsize-{self.patch_size}_step-{self.step}.npy")
                    if os.path.exists(path2coord):
                        coord = np.load(path2coord, mmap_mode="r")
                    else:
                        coord = self.get_coord(test=t)
                    img_shape = self.manager.dataset[t_name][2:]
                    occluded_prob = {}
                    start = time.time()
                    cnt = 0
                    for x, y, z in coord:            
                        occluded_prob[f"({x}, {y}, {z})"] = []
                        self.manager.dataset[t_name].transforms = Compose([Cutout(self.patch_size, random_size=False, 
                                                                           localization=(x, y, z), img_shape=img_shape)])
                        loader = self.manager.get_dataloader(test=(t == "external"),
                                                            test_intra=(t == "internal"),
                                                            run_index=run)
                        model.model.eval()
                        pbar = tqdm(total=len(loader.test), desc=f"({x}, {y}, {z})")
                        with torch.no_grad():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                            for dataitem in loader.test:
                                pbar.update()
                                inputs = dataitem.inputs
                                if isinstance(inputs, torch.Tensor):
                                    inputs = inputs.to(torch.device("cuda" if self.args.cuda 
                                                                else "cpu"))
                                outputs = model.model(inputs)
                                outputs = torch.sigmoid(outputs)
                                occluded_prob[f"({x}, {y}, {z})"].extend(outputs.cpu().detach().numpy())
                        pbar.close()
                        occluded_prob[f"({x}, {y}, {z})"] = np.asarray(occluded_prob[f"({x}, {y}, {z})"])
                        cnt += 1
                    
                    logger.debug(f"Occluded Probability : {occluded_prob[f'({x}, {y}, {z})'].shape}")
                    logger.debug(f"Number of patch : {cnt} | Time : {time.time() - start}")
                    relevance_maps[t] = {coord: (unoccluded_prob - occ_prob)
                                         for coord, occ_prob in occluded_prob.items()}
                    relevance_maps["patch_size"] = self.patch_size
                    relevance_maps["step"] = self.step
                    relevance_maps["img_size"] = img_shape
                with open(os.path.join(self.args.checkpoint_dir, pck_name), 'wb') as f:
                    pickle.dump(relevance_maps, f)    
                logger.debug(f"Pickle saved to : {os.path.join(self.args.checkpoint_dir, pck_name)}")
                logger.debug(f"Duration : {time.time() - start}")
                
    def get_runs_to_test(self):
        if self.args.runs is not None and len(self.args.runs) > 0:
            runs = self.args.runs
        else:
            runs = list(range(self.args.nb_runs))
        return runs

    def get_epochs_to_test(self):
        # Get the last point and tests it, for each run
        epochs_tested = [[self.args.nb_epochs - 1] for _ in range(self.args.nb_runs)]

        return epochs_tested
    
    def get_coord(self, test):
        logger.info("Get coordinates where there are skeleton values.")
        coord = []
        arr_skel = self.manager.dataset[test].get_data()
        arr_skel = arr_skel.sum(axis=(0,1))
        for x, y, z in [[i for i in range(self.patch_size, arr_skel.shape[j], self.step)] for j in range(3)]:
            if np.count_nonzero(arr_skel[x-self.patch_size:x+self.patch_size,
                                         y-self.patch_size:y+self.patch_size,
                                         z-self.patch_size:z+self.patch_size]) > 0:
                coord.append([x,y,z])
        coord = np.asarray(coord)
        path2coord = os.path.join(self.args.root, f"coordinates_{self.args.pb}_patchsize-{self.patch_size}_step-{self.step}.npy")
        np.save(path2coord, coord)
        logger.info(f"Save coordinates at: {path2coord}")
        return coord


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Creating saliency maps with patch occlusions.')

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

    # Depends on available CPU/GPU memory
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Number of images in the mini-batch. Default is: 32")
    parser.add_argument("--nb_epochs", type=int, default=50,
                        help="Last training epoch. Default is: 50.")

    # Dataloader: set them
    parser.add_argument("--num_cpu_workers", type=int, default=3,
                        help="Number of workers assigned to do the preprocessing step (used by DataLoader of Pytorch). "
                        "Default is: 3")
    
    # This code can be executed on CPU or GPU
    parser.add_argument("--cuda", type=bool, default=True, 
                        help="If True, executes the code on GPU. Default is: True.")

    # Parameters of patfch occlusion
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for occlusions. Default is: 16.")
    parser.add_argument("--step", type=int, default=4, help="Step between between patches. Default is: 4.")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate verbosity mode")

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    # Setup Logging
    setup_logging(level="debug" if args.verbose else "info",
                  logfile=os.path.join(args.checkpoint_dir, f"exp-{args.exp_name}.log"))
    
    logger.info(f"Checkpoint directory : {args.checkpoint_dir}")

    tester = PatchOcclusion(args)
    tester.run()
    

if __name__ == "__main__":
    main(sys.argv[1:])
