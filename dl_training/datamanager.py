import numpy as np
import pandas as pd
import torch
import logging
from collections import namedtuple

from torchvision.transforms.transforms import Compose
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from datasets.open_bhb import OpenBHB, SubOpenBHB, List
from datasets.bhb_10k import BHB
from datasets.clinical_multisites import SCZDataset, BipolarDataset, ASDDataset, SubSCZDataset, \
    SubBipolarDataset, SubASDDataset
from contrastive_learning.contrastive_datasets import ContrastiveSCZDataset, \
    ContrastiveBipolarDataset, ContrastiveASDDataset, ContrastiveOpenBHB, ContrastiveSubOpenBHB
from preprocessing.transforms import Padding, Crop, Normalize, Binarize
from augmentation.da_module import DA_Module

logger = logging.getLogger()
SetItem = namedtuple("SetItem", ["test", "train", "validation"], defaults=(None,) * 3)
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels"])


class OpenBHBDataManager:

    def __init__(self, root: str, preproc: str, labels: List[str] = None, sampler: str = "random",
                 batch_size: int = 1, number_of_folds: int = None, N_train_max: int = None,
                 mask=None, model: str = None, device: str = "cuda", **dataloader_kwargs):
        assert model in [None, "base", "SimCLR", "SupCon", "y-aware"], f"Unknown model {model}"
        assert sampler in ["random", "sequential"], "Unknown sampler '%s'" % sampler
        self.dataset = dict()
        self.labels = labels or []
        self.mask = mask
        scheme = "train_val_test"
        self.scheme = scheme

        input_transforms = self.get_input_transforms(preproc=preproc, model=model)

        if N_train_max is not None:
            logger.info("Automatic stratification on Age+Sex+Site")
            if model in ["SimCLR", "SupCon", "y-aware"]:
                dataset_cls = ContrastiveSubOpenBHB
            else:
                dataset_cls = SubOpenBHB
            self.dataset["train"] = [dataset_cls(root, preproc=preproc, scheme=scheme,
                                                 split="train", stratify=True,
                                                 transforms=input_transforms,
                                                 N_train_max=N_train_max, nb_folds=number_of_folds,
                                                 fold=fold, target=labels)
                                     for fold in range(number_of_folds)]
            self.dataset["validation"] = [dataset_cls(root, preproc=preproc, scheme=scheme, split="val",
                                                      transforms=input_transforms, target=labels)
                                          for _ in range(number_of_folds)]
            self.number_of_folds = number_of_folds
        else:
            kwargs_dataset = dict()
            if model in ["SimCLR", "SupCon", "y-aware"]:
                dataset_cls = ContrastiveOpenBHB
            else:
                dataset_cls = OpenBHB
            self.dataset["train"] = [dataset_cls(root, preproc=preproc, scheme=scheme, split="train",
                                                 transforms=input_transforms, target=labels,
                                                 **kwargs_dataset)]
            self.dataset["validation"] = [dataset_cls(root, preproc=preproc, scheme=scheme, split="val",
                                                      transforms=input_transforms, target=labels,
                                                      **kwargs_dataset)]
            self.number_of_folds = 1
        if model in ["SimCLR", "SupCon", "y-aware"]:
            dataset_cls = ContrastiveOpenBHB
        else:
            dataset_cls = OpenBHB
        self.dataset["test"] = dataset_cls(root, preproc=preproc, scheme=scheme, split="test",
                                           transforms=input_transforms, target=labels)
        self.dataset["test_intra"] = dataset_cls(root, preproc=preproc, scheme=scheme, split="test_intra",
                                                 transforms=input_transforms, target=labels)
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.dataloader_kwargs = dataloader_kwargs

    @staticmethod
    def collate_fn(list_samples):
        """ After fetching a list of samples using the indices from sampler,
              the function passed as the collate_fn argument is used to collate lists
              of samples into batches.

              A custom collate_fn is used here to apply the transformations.

              See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn.
              """
        data = dict(outputs=None)  # compliant with DataManager <collate_fn>
        # data["inputs"] = torch.stack([torch.from_numpy(sample[0]) for sample in list_samples], dim=0).float()
        data["inputs"] = torch.stack([torch.from_numpy(np.copy(sample[0])) for sample in list_samples], dim=0).float()
        data["labels"] = torch.stack([torch.tensor(sample[1]) for sample in list_samples], dim=0).squeeze().float()
        return DataItem(**data)

    def get_dataloader(self, train=False, validation=False,
                       test=False, test_intra=False, fold_index=None):

        assert test + test_intra <= 1, "Only one tests accepted"
        _test, _train, _validation, sampler = (None, None, None, None)
        tests_to_return = []
        if validation:
            tests_to_return.append("validation")
        if test:
            tests_to_return.append("test")
        if test_intra:
            tests_to_return.append("test_intra")
        test_loaders = dict()
        for t in tests_to_return:
            dataset = self.dataset[t] if t != "validation" else self.dataset[t][fold_index]
            drop_last = True if len(dataset) % self.batch_size == 1 else False
            if drop_last and t != "validation":
                logger.warning(f"The last subject will not be tested ! "
                               f"Change the batch size ({self.batch_size}) to test on all subject ({len(dataset)})")
            test_loaders[t] = DataLoader(dataset, batch_size=self.batch_size,
                                         collate_fn=OpenBHBDataManager.collate_fn, drop_last=drop_last,
                                         **self.dataloader_kwargs)
        if "test_intra" in test_loaders:
            assert "test" not in test_loaders
            test_loaders["test"] = test_loaders.pop("test_intra")
        if train:
            if self.sampler == "random":
                sampler = RandomSampler(self.dataset["train"][fold_index])
            elif self.sampler == "sequential":
                sampler = SequentialSampler(self.dataset["train"][fold_index])
            dataset = self.dataset["train"][fold_index]
            drop_last = True if len(dataset) % self.batch_size == 1 else False
            _train = DataLoader(
                dataset, batch_size=self.batch_size, sampler=sampler,
                collate_fn=OpenBHBDataManager.collate_fn, drop_last=drop_last,
                **self.dataloader_kwargs)

        return SetItem(train=_train, **test_loaders)

    @staticmethod
    def get_input_transforms(preproc, model="base", data_augmentation=None):
        if preproc in ["vbm", "quasi_raw"]:
            # Input size 121 x 145 x 121
            input_transforms = Compose(
                [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'), Normalize()])
        elif preproc in ["skeleton"]:
            # Input size 128 x 152 x 128
            input_transforms = Compose(
                [Padding([1, 128, 160, 128], mode='constant'), Binarize(one_values=[30, 35, 60, 100, 110, 120])])
        else:
            raise ValueError("Unknown preproc: %s" % preproc)
        if model in ["SimCLR", "SupCon", "y-aware"]:
            if data_augmentation is None:
                input_transforms.transforms.append(DA_Module())
                logger.info("Data augmentation is set to the standard DA_model")
            else:
                input_transforms.transforms.append(DA_Module(transforms=data_augmentation))
        return input_transforms

    def get_nb_folds(self):
        return self.number_of_folds


class BHBDataManager(OpenBHBDataManager):

    def __init__(self, root: str, preproc: str, labels: List[str] = None, sampler: str = "random", batch_size: int = 1,
                 number_of_folds: int = None, N_train_max: int = None, mask=None,
                 model: str = None, device: str = "cuda", scheme: str = "train_val_test", **dataloader_kwargs):
        super().__init__(root, preproc, labels, sampler, batch_size, number_of_folds, N_train_max, mask,
                         model, device, **dataloader_kwargs)
        assert model in [None, "base"], "Unknown model: %s" % model
        assert sampler in ["random", "sequential"], "Unknown sampler '%s'" % sampler
        assert scheme == "train_val_test", "Scheme %s not implemented yet" % scheme
        assert N_train_max is None, "Sub-sampling BHB not implemented yet"

        self.dataset = dict()
        self.labels = labels or []
        self.mask = mask
        self.number_of_folds = number_of_folds
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.dataloader_kwargs = dataloader_kwargs
        self.scheme = scheme

        input_transforms = self.get_input_transforms(preproc=preproc)

        self.dataset["train"] = [BHB(root, preproc=preproc, scheme=self.scheme, split="train",
                                     transforms=input_transforms, target=labels)
                                 for _ in range(self.number_of_folds)]
        self.dataset["validation"] = [BHB(root, preproc=preproc, scheme=self.scheme, split="val",
                                          transforms=input_transforms, target=labels)
                                      for _ in range(self.number_of_folds)]
        self.dataset["test"] = BHB(root, preproc=preproc, scheme=self.scheme, split="test",
                                   transforms=input_transforms, target=labels)
        self.dataset["test_intra"] = BHB(root, preproc=preproc, scheme=self.scheme, split="test_intra",
                                         transforms=input_transforms, target=labels)


class ClinicalDataManager(OpenBHBDataManager):

    def __init__(self, root: str, preproc: str, db: str, labels: List[str] = None, sampler: str = "random",
                 batch_size: int = 1, number_of_folds: int = None, N_train_max: int = None,
                 mask=None, model: str = "base", data_augmentation: List[str] = None,
                 device: str = "cuda", **dataloader_kwargs):

        assert db in ["scz", "bipolar", "asd"], "Unknown db: %s" % db
        assert sampler in ["random", "sequential"], "Unknown sampler '%s'" % sampler
        assert model in [None, "base", "SimCLR", "SupCon", "y-aware"], f"Unknown model {model}"

        self.dataset = dict()
        self.labels = labels or []
        self.mask = mask
        self.number_of_folds = number_of_folds
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.dataloader_kwargs = dataloader_kwargs

        input_transforms = self.get_input_transforms(preproc=preproc, model=model,
                                                     data_augmentation=data_augmentation)
        logger.debug(f"input_transforms : {input_transforms}")
        dataset_cls = None
        if db == "scz":
            if model in ["SimCLR", "SupCon", "y-aware"]:
                dataset_cls = ContrastiveSCZDataset
            else:
                dataset_cls = SCZDataset if N_train_max is None else SubSCZDataset
        elif db == "bipolar":
            if model in ["SimCLR", "SupCon", "y-aware"]:
                dataset_cls = ContrastiveBipolarDataset
            else:
                dataset_cls = BipolarDataset if N_train_max is None else SubBipolarDataset
        elif db == "asd":
            if model in ["SimCLR", "SupCon", "y-aware"]:
                dataset_cls = ContrastiveASDDataset
            else:
                dataset_cls = ASDDataset if N_train_max is None else SubASDDataset
        logger.debug(f"Dataset CLS : {dataset_cls}")
        if N_train_max is None:
            self.dataset["train"] = [dataset_cls(root, preproc=preproc, split="train",
                                                 transforms=input_transforms, target=labels)
                                     for _ in range(self.number_of_folds)]
        else:
            self.dataset["train"] = [dataset_cls(root, N_train_max=N_train_max, fold=f,
                                                 nb_folds=self.number_of_folds, preproc=preproc, split="train",
                                                 transforms=input_transforms, target=labels)
                                     for f in range(self.number_of_folds)]
        self.dataset["validation"] = [dataset_cls(root, preproc=preproc, split="val",
                                                  transforms=input_transforms, target=labels)
                                      for _ in range(self.number_of_folds)]
        self.dataset["test"] = dataset_cls(root, preproc=preproc, split="test",
                                           transforms=input_transforms, target=labels)
        self.dataset["test_intra"] = dataset_cls(root, preproc=preproc, split="test_intra",
                                                 transforms=input_transforms, target=labels)