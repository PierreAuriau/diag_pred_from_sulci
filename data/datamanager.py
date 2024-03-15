# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import logging  
from collections import namedtuple
from typing import Callable, List, Type, Sequence, Dict

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from data.datasets import SCZDataset, BDDataset, ASDDataset
from contrastive_learning.contrastive_datasets import ContrastiveSCZDataset, \
    ContrastiveBDDataset, ContrastiveASDDataset
from img_processing.da_module import DAModule

logger = logging.getLogger()
SetItem = namedtuple("SetItem", ["test", "train", "validation"], defaults=(None,) * 3)
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels"])


class ClinicalDataManager(object):

    def __init__(self, root: str, preproc: str, db: str, labels: List[str] = None, sampler: str = "random",
                 batch_size: int = 1, nb_runs: int = None, model: str = "base", data_augmentation: List[str] = None,
                 device: str = "cuda", **dataloader_kwargs):

        assert db in ["scz", "bd", "asd"], "Unknown db: %s" % db
        assert sampler in ["random", "sequential"], "Unknown sampler '%s'" % sampler
        assert model in [None, "base", "SimCLR", "SupCon", "y-aware"], f"Unknown model {model}"
        assert preproc in [None, "no", "smoothing"]

        self.dataset = dict()
        self.labels = labels or []
        self.nb_runs = nb_runs
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.dataloader_kwargs = dataloader_kwargs

        if data_augmentation is None:
            data_augmentation = ["no"]
        data_augmentation = DAModule(transforms=data_augmentation)
        logger.debug(f"data_augmentation : {data_augmentation}")
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
        logger.debug(f"Dataset CLS : {dataset_cls.__name__}")
        dataset = dataset_cls(root, split="train", preproc=preproc,
                              transforms=data_augmentation, target=labels)
        self.dataset["train"] = [dataset for _ in range(self.nb_runs)]
        dataset = dataset_cls(root, split="val", preproc=preproc, target=labels)
        self.dataset["validation"] = [dataset for _ in range(self.nb_runs)]
        for split in ["test", "test_intra"]:
            dataset = dataset_cls(root, split=split, preproc=preproc, target=labels)
            self.dataset[split] = dataset

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
                       test=False, test_intra=False, run=None):

        assert test + test_intra <= 1, "Only one tests accepted"
        tests_to_return = []
        if validation:
            tests_to_return.append("validation")
        if test:
            tests_to_return.append("test")
        if test_intra:
            tests_to_return.append("test_intra")
        test_loaders = dict()
        for t in tests_to_return:
            dataset = self.dataset[t] if t != "validation" else self.dataset[t][run]
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
                sampler = RandomSampler(self.dataset["train"][run])
            elif self.sampler == "sequential":
                sampler = SequentialSampler(self.dataset["train"][run])
            dataset = self.dataset["train"][run]
            drop_last = True if len(dataset) % self.batch_size == 1 else False
            _train = DataLoader(
                dataset, batch_size=self.batch_size, sampler=sampler,
                collate_fn=self.collate_fn, drop_last=drop_last,
                **self.dataloader_kwargs)
        else:
            _train = None
        return SetItem(train=_train, **test_loaders)

    def get_number_of_runs(self):
        return self.nb_runs

    def __str__(self):
        return "ClinicalDataManager"
