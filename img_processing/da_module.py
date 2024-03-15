# -*- coding: utf-8 -*-

from img_processing.transformer import Transformer
from img_processing.augmentations import Cutout, Rotation


class DAModule(object):
    def __init__(self, transforms=("Rotation", "Cutout")):
        self.compose_transforms = Transformer()
        for t in transforms:
            if t == "Cutout":
                self.compose_transforms.register(Cutout(patch_size=0.4, random_size=True,
                                                        localization="on_data", min_size=0.1),
                                                 probability=1, with_channel=True)
            elif t == "Rotation":
                self.compose_transforms.register(Rotation(angles=5, axes=[(0,1), (0,2), (1,2)], order=0),
                                                 probability=0.5, with_channel=True)
            elif t == "Rotation_Cutout":
                # for wandb
                self.compose_transforms.register(Cutout(patch_size=0.4, random_size=True,
                                                        localization="random", min_size=0.1),
                                                 probability=1, with_channel=True)
                self.compose_transforms.register(Rotation(angles=5, axes=[(0,1), (0,2), (1,2)], order=0),
                                                 probability=0.5, with_channel=True)
            elif t == "no":
                # no da
                break
            else:
                raise ValueError(f"Unknown data augmentation : {t}")

    def __call__(self, x):
        return self.compose_transforms(x)

    def __str__(self):
        if len(self.compose_transforms) == 0:
            return "Empty DAModule"
        string = "DAModule :"
        for trf in self.compose_transforms.transforms:
            string += f"\n\t* {trf} | p={trf.probability}"
        return string
