# -*- coding: utf-8 -*-

import numpy as np
from datasets.clinical_multisites import SCZDataset, ASDDataset, BipolarDataset


class ContrastiveSCZDataset(SCZDataset):

    def __getitem__(self, idx: int):
        np.random.seed()
        x1, y1 = super().__getitem__(idx)
        x2, y1 = super().__getitem__(idx)
        return np.stack((x1, x2), axis=0), y1


class ContrastiveASDDataset(ASDDataset):

    def __getitem__(self, idx: int):
        np.random.seed()
        x1, y1 = super().__getitem__(idx)
        x2, y1 = super().__getitem__(idx)
        return np.stack((x1, x2), axis=0), y1


class ContrastiveBipolarDataset(BipolarDataset):

    def __getitem__(self, idx: int):
        np.random.seed()
        x1, y1 = super().__getitem__(idx)
        x2, y1 = super().__getitem__(idx)
        return np.stack((x1, x2), axis=0), y1
