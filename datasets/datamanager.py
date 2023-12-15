# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import logging  
from collections import namedtuple
from typing import Callable, List, Type, Sequence, Dict

from torchvision.transforms.transforms import Compose
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from datasets.clinical_multisites import SCZDataset, BDDataset, ASDDataset
from contrastive_learning.contrastive_datasets import ContrastiveSCZDataset, \
    ContrastiveBDDataset, ContrastiveASDDataset
from img_processing.preprocessing import Padding, Crop, Normalize, Binarize
from img_processing.da_module import DAModule

logger = logging.getLogger()
SetItem = namedtuple("SetItem", ["test", "train", "validation"], defaults=(None,) * 3)
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels"])


class ClinicalDataManager(object):

    def __init__(self, root: str, preproc: str, db: str, labels: List[str] = None, sampler: str = "random",
                 batch_size: int = 1, number_of_trainings: int = None, mask=None,
                 model: str = "base", data_augmentation: List[str] = None,
                 device: str = "cuda", **dataloader_kwargs):

        assert db in ["scz", "bd", "asd"], "Unknown db: %s" % db
        assert sampler in ["random", "sequential"], "Unknown sampler '%s'" % sampler
        assert model in [None, "base", "SimCLR", "SupCon", "y-aware"], f"Unknown model {model}"

        self.dataset = dict()
        self.labels = labels or []
        self.mask = mask
        self.number_of_trainings = number_of_trainings
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.dataloader_kwargs = dataloader_kwargs

        input_transforms = self.get_input_transforms(preproc=preproc, model=model,
                                                     data_augmentation=data_augmentation)
        logger.debug(f"input_transforms : {input_transforms}")
        dataset_cls = None
        if db == "scz":
            if model == "SupCon":
                dataset_cls = ContrastiveSCZDataset
            else:
                dataset_cls = SCZDataset
        elif db == "bd":
            if model == "SupCon":
                dataset_cls = ContrastiveBDDataset
            else:
                dataset_cls = BDDataset
        elif db == "asd":
            if model == "SupCon":
                dataset_cls = ContrastiveASDDataset
            else:
                dataset_cls = ASDDataset
        logger.debug(f"Dataset CLS : {dataset_cls}")
        self.dataset["train"] = [dataset_cls(root, preproc=preproc, split="train",
                                             transforms=input_transforms, target=labels)
                                 for _ in range(self.number_of_trainings)]
        self.dataset["validation"] = [dataset_cls(root, preproc=preproc, split="val",
                                                  transforms=input_transforms, target=labels)
                                      for _ in range(self.number_of_trainings)]
        self.dataset["test"] = dataset_cls(root, preproc=preproc, split="test",
                                           transforms=input_transforms, target=labels)
        self.dataset["test_intra"] = dataset_cls(root, preproc=preproc, split="test_intra",
                                                 transforms=input_transforms, target=labels)

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
        data["inputs"] = torch.stack([torch.from_numpy(np.copy(sample[0])) for sample in list_samples],
                                     dim=0).float()
        data["labels"] = torch.stack([torch.tensor(sample[1]) for sample in list_samples], dim=0).squeeze().float()
        return DataItem(**data)

    def get_dataloader(self, train=False, validation=False,
                       test=False, test_intra=False, training_index=None):

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
            dataset = self.dataset[t] if t != "validation" else self.dataset[t][training_index]
            drop_last = True if len(dataset) % self.batch_size == 1 else False
            if drop_last and t != "validation":
                logger.warning(f"The last subject will not be tested ! "
                               f"Change the batch size ({self.batch_size}) to test on all subject ({len(dataset)})")
            test_loaders[t] = DataLoader(dataset, batch_size=self.batch_size,
                                         collate_fn=self.collate_fn, drop_last=drop_last,
                                         **self.dataloader_kwargs)
        if "test_intra" in test_loaders:
            assert "test" not in test_loaders
            test_loaders["test"] = test_loaders.pop("test_intra")
        if train:
            if self.sampler == "random":
                sampler = RandomSampler(self.dataset["train"][training_index])
            elif self.sampler == "sequential":
                sampler = SequentialSampler(self.dataset["train"][training_index])
            dataset = self.dataset["train"][training_index]
            drop_last = True if len(dataset) % self.batch_size == 1 else False
            _train = DataLoader(
                dataset, batch_size=self.batch_size, sampler=sampler,
                collate_fn=self.collate_fn, drop_last=drop_last,
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
        if model == "SupCon":
            if data_augmentation is None:
                input_transforms.transforms.append(DAModule())
                logger.info("Data augmentation is set to the standard DA_model")
            else:
                input_transforms.transforms.append(DAModule(transforms=data_augmentation))
        return input_transforms

    def get_nb_trainings(self):
        return self.number_of_trainings

    def __str__(self):
        return "ClinicalDataManager"

