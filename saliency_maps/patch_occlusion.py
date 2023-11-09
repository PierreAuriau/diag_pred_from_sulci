import os
import logging
import argparse
import pickle
import nibabel as nib
import numpy as np
from tqdm import tqdm
import time

import torch
from torchvision.transforms.transforms import Compose

from logs.utils import setup_logging, get_chk_name
from dl_training.core import Base
from dl_training.training import BaseTrainer
from transformations.augmentations import Cutout
from transformations.preprocessing import Padding, Binarize

logger = logging.getLogger()


class PatchOcclusion():

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.model, num_classes=1, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args.model, args.pb, args.preproc, args.root, args.N_train_max,
                                                      sampler=args.sampler, batch_size=args.batch_size,
                                                      number_of_folds=args.nb_folds, data_augmentation=args.data_augmentation,
                                                      device=('cuda' if args.cuda else 'cpu'),
                                                      num_workers=args.num_cpu_workers,
                                                      pin_memory=True)
        self.loss = BaseTrainer.build_loss(args.model, args.pb, args.cuda)
        self.metrics = BaseTrainer.build_metrics(args.pb, args.model)
        self.kwargs_test = dict()

        if self.args.pretrained_path and self.manager.number_of_folds > 1:
            logger.warning('Several folds found while a unique pretrained path is set!')

    def run(self):
        if self.args.outfile_name is None:
            self.args.outfile_name = f"PatchOcclusion_{self.args.exp_name}"
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                logger.debug(f"Pretrained path : {pretrained_path}")

                exp_name = f"{self.args.outfile_name}_fold{fold}_epoch{epoch}.pkl"

                relevance_maps = {}
                for t in ["internal"]:
                    
                    loader = self.manager.get_dataloader(test=(t == "external"),
                                                         test_intra=(t == "internal"),
                                                         fold_index=fold)
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
                    patch_size = 16
                    step = 4
                    img_shape = [128, 160, 128]
                    occluded_prob = {}
                    start = time.time()
                    cnt = 0
                    coord = np.load(f"{self.args.pb}_coordinates_patchsize{patch_size}_step{step}.npy", mmap_mode="r")
                    for x, y, z in coord:            
                        occluded_prob[f"({x}, {y}, {z})"] = []
                        t_name = {"internal": "test_intra", "external": "test"}[t]
                        self.manager.dataset[t_name].transforms = Compose([Padding([1] + img_shape, mode='constant'), 
                                                                        Binarize(threshold=0),
                                                                        Cutout(patch_size, random_size=False, 
                                                                            localization=(x, y, z), img_shape=img_shape)])
                        loader = self.manager.get_dataloader(test=(t == "external"),
                                                            test_intra=(t == "internal"),
                                                            fold_index=fold)
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
                    relevance_maps["patch_size"] = patch_size
                    relevance_maps["step"] = step
                    relevance_maps["img_size"] = img_shape
                with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                    pickle.dump(relevance_maps, f)    
                logger.debug(f"Pickle saved to : {os.path.join(self.args.checkpoint_dir, exp_name)}")
                logger.debug(f"Duration : {time.time() - start}")
                
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pb", type=str, choices=["scz", "bipolar", "asd"], required=True)
    args = parser.parse_args()

    args.root = "/home_local/pa267054/root"
    args.preproc = "skeleton"
    args.net = "densenet121"
    args.model = "base"
    args.batch_size = 32
    args.nb_folds = 3
    args.sampler = "sequential"
    args.num_cpu_workers = 8
    args.cuda = True
    args.verbose = True
    args.folds = [0]
    args.data_augmentation = None
    #args.outfile_name = None
    args.sigma = 0
    args.pretrained_path = None
    #if args.pb in ["age", "sex"]:
    #    args.N_train_max = 1100
    #else:
    #    args.N_train_max = None
    args.N_train_max = None
    args.lr = 1e-4   

    target2chkpt = {
        "asd": "20230530_densenet_skel_asd",
        "bipolar": "20230531_densenet_skel_bip",
        "scz": "20230111_schizophrenic_wo_ventricle_skeletons"
    }

    target2model = {
        "asd": "",
        "bipolar": "clean-sweep-1",
        "scz": "sleek-sweep-2"
    }

    target2fold = {"bipolar": 0, "asd": 1, "scz": 2}

    target = args.pb
    chkpt = target2chkpt[target]
    model = target2model[target]
    args.checkpoint_dir = os.path.join("/neurospin/psy_sbox/analyses/202205_predict_neurodev/models", chkpt, model)
    args.exp_name = f"densenet121_skeleton_{target}" if target in ["asd", "bipolar"] else "scz_skeleton_densenet121"
    args.nb_epochs = 50 if target in ["asd", "bipolar"] else 100
    args.outfile_name = f"PatchOcclusionVF_{args.exp_name}"
    args.folds = [target2fold[target]]

    # Setup Logging
    setup_logging(level="debug" if args.verbose else "info",
                    logfile=os.path.join(args.checkpoint_dir, f"{args.exp_name}.log"))
    logger.info(f"Target {target}")
    logger.info(f"Checkpoint directory : {args.checkpoint_dir}")

    tester = PatchOcclusion(args)
    tester.run()
    
    
    """
    fold = 0
    epoch = 49
    tester = SulcusOcclusion(args)
    pretrained_path = tester.args.pretrained_path or \
                                  os.path.join(tester.args.checkpoint_dir, get_chk_name(tester.args.exp_name, fold, epoch))
    logger.debug(f"Pretrained path : {pretrained_path}")
    tests = ["internal", "external"]
                
    for t in tests:           
        
        loader = tester.manager.get_dataloader(test=(t == "external"),
                                                test_intra=(t == "internal"),
                                                fold_index=fold)
        model = Base(model=tester.net, loss=tester.loss,
                        metrics=tester.metrics,
                        pretrained=pretrained_path,
                        use_cuda=tester.args.cuda)
        
        

        patch_size = [64, 80, 64]
        img_shape = [128, 160, 128]
        for x in range(patch_size[0]//2, img_shape[0], patch_size[0]):
            for y in range(patch_size[1]//2, img_shape[1], patch_size[1]):
                for z in range(patch_size[2]//2, img_shape[2], patch_size[2]):
                    t_name = {"internal": "test_intra", "external": "test"}[t]
                    tester.manager.dataset[t_name].transforms = Compose([Padding([1] + img_shape, mode='constant'), 
                                                                 Binarize(threshold=0),
                                                                 Cutout(patch_size, random_size=False, 
                                                                        localization=(x, y, z), img_shape=img_shape)])
                    loader = tester.manager.get_dataloader(test=(t == "external"),
                                                           test_intra=(t == "internal"),
                                                           fold_index=fold)
                    for dataitem in loader.test:
                        inputs = dataitem.inputs
                        img = inputs.cpu().detach().numpy()
                        img = img.squeeze()
                        print(np.unique(img))
                        print(np.count_nonzero(img))
                        skel = nib.load("/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/cnp/1.5mm/wo_ventricles/F/Fresampled_skeleton_sub-11149_ses-1.nii.gz")
                        ni_img = nib.Nifti1Image(img, skel.affine)
                        nib.save(ni_img, f"/neurospin/dico/pauriau/scripts/outputs/input_patchocclusion_{x}_{y}_{z}.nii.gz")
                        break    

        import pdb 
        pdb.set_trace()
        print(y)
        print(len(y))
        print(type(y))
        tester.manager.dataset["test"].transforms = 
        tester.manager.dataset["test"].transforms = Compose([Padding([1, 160, 160, 160], mode='constant'), Binarize(threshold=0)])
        for dataitem in loader.test: print(np.count_nonzero(dataitem.inputs.detach().cpu().numpy()))
        """