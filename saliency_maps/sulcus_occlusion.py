import os
import pickle
import nibabel
from logs.utils import get_chk_name
from dl_training.core import Base
from dl_training.training import BaseTrainer
import logging
import argparse
from logs.utils import setup_logging

logger = logging.getLogger()


class SulcusOcclusion():

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
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                logger.debug(f"Pretrained path : {pretrained_path}")
                tests = ["internal", "external"]
                saliency_maps = {t:[] for t in tests}
                for t in tests:
                    if self.args.outfile_name is None:
                        self.args.outfile_name = f"SulcusOcclusion_{test}test_{self.args.exp_name}"
                    exp_name = f"{self.args.outfilename}_fold{fold}_epoch{epoch}"
                    # Get sulcus label instead of skeletons
                    loader = self.manager.get_dataloader(test=(t == "external"),
                                                         test_intra=(t == "internal"),
                                                         fold_index=fold)
                    model = Base(model=self.net, loss=self.loss,
                                 metrics=self.metrics,
                                 pretrained=pretrained_path,
                                 use_cuda=self.args.cuda)
                    
                    # Probabilities without occlusions
                    unoccluded_prob = []
                    model.eval()
                    pbar = tqdm(total=len(loader.test), desc="Mini-Batch")
                    with torch.no_grad():
                        for dataitem in loader:
                            pbar.update()
                            inputs = dataitem.inputs
                            if isinstance(inputs, torch.Tensor):
                                inputs = inputs.to(self.device)
                            outputs = self.model(inputs)
                            outputs = torch.sigmoid(outputs)
                    unoccluded_prob.extend(outputs.cpu().detach().numpy())
                    pbar.close()
                    unoccluded_prob = np.asarray(unoccluded_prob)

                    # Probabilities with occlusions
                    # For occlusion, input_transform = Binarize, value = sulci
                    # get sulcuslabel instead of skeletons
                    # get dico_sulci
                    occluded_prob = {}
                    for label, value in dico_sulci.items():
                        occluded_prob[label] = []
                        t_name = {"internal": "test_intra", "external": "test"}[t]
                        self.manager.dataset["t"].transforms = Compose([Padding([1, 128, 160, 128], mode='constant'), 
                                                                        Binarize(one_values=[value])])
                        loader = self.manager.get_dataloader(test=(t == "external"),
                                                            test_intra=(t == "internal"),
                                                            fold_index=fold)
                        model.eval()
                        pbar = tqdm(total=len(loader.test), desc="Mini-Batch")
                        with torch.no_grad():
                            for dataitem in loader:
                                pbar.update()
                                inputs = dataitem.inputs
                                if isinstance(inputs, torch.Tensor):
                                    inputs = inputs.to(self.device)
                                outputs = self.model(inputs)
                                outputs = torch.sigmoid(outputs)
                        occluded_prob[label].extend(outputs.cpu().detach().numpy())
                        pbar.close()
                        occluded_prob[label] = np.asarray(occluded_prob[label])

                    relevance_maps = {label: (unoccluded_prob - occ_prob)
                                      for label, occ_prob in unoccluded_prob.items()}

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
    args =  argparse.ArgumentParser().parse_args()
    args.root = "/home_local/pa267054/root"
    args.preproc = "skeleton"
    args.checkpoint_dir = "/neurospin/psy_sbox/analyses/202205_predict_neurodev/models/20230531_densenet_skel_bip"
    args.exp_name = "densenet121_skeleton_bipolar"
    args.pb = "bipolar"
    args.net = "densenet121"
    args.model = "base"
    args.batch_size = 32
    args.nb_epochs = 50
    args.nb_folds = 3
    args.sampler = "sequential"
    args.num_cpu_workers = 8
    args.cuda = True
    args.verbose = True
    args.folds = None
    args.data_augmentation = None
    args.outfile_name = None
    args.sigma = 0
    args.pretrained_path = None
    if args.pb in ["age", "sex"]:
        args.N_train_max = 1100
    else:
        args.N_train_max = None
    args.lr = 1e-4

    # Setup Logging
    setup_logging(level="debug" if args.verbose else "info",
                  logfile=os.path.join(args.checkpoint_dir, f"{args.exp_name}.log"))
    logger.info(f"Checkpoint directory : {args.checkpoint_dir}")

    fold = 0
    epoch = 49
    tester = SulcusOcclusion(args)
    pretrained_path = tester.args.pretrained_path or \
                                  os.path.join(tester.args.checkpoint_dir, get_chk_name(tester.args.exp_name, fold, epoch))
    logger.debug(f"Pretrained path : {pretrained_path}")
    tests = ["internal", "external"]
                
    for t in tests:           
        # Get sulcus label instead of skeletons
        loader = tester.manager.get_dataloader(test=(t == "external"),
                                                test_intra=(t == "internal"),
                                                fold_index=fold)
        model = Base(model=tester.net, loss=tester.loss,
                        metrics=tester.metrics,
                        pretrained=pretrained_path,
                        use_cuda=tester.args.cuda)
        
        

        y, y_true, _, _, _ = model.test(loader.test, set_name=f"{t} test")

        import pdb 
        pdb.set_trace()

        print(y)
        print(len(y))
        print(type(y))
        tester.manager.dataset["test"].transforms = 
        tester.manager.dataset["test"].transforms = Compose([Padding([1, 160, 160, 160], mode='constant'), Binarize(threshold=0)])
        for dataitem in loader.test: print(np.count_nonzero(dataitem.inputs.detach().cpu().numpy()))