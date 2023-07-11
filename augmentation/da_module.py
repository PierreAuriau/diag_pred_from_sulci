from augmentation import Transformer
from augmentation.intensity import add_blur, add_noise
from augmentation.spatial import flip, cutout, cutout_with_threshold, rotation, random_cutout
from preprocessing.transforms import Crop


class DA_Module(object):

    def __init__(self, transforms=("cutout", "rotation")):
        self.compose_transforms = Transformer()
        for t in transforms:
            if t == "flip":
                self.compose_transforms.register(flip, probability=0.5)
            if t == "add_blur":
                self.compose_transforms.register(add_blur, probability=0.5, sigma=(0.1, 1))
            if t == "add_noise":
                self.compose_transforms.register(add_noise, sigma=(0.1, 1), probability=0.5)
            if t == "cutout":
                self.compose_transforms.register(cutout, probability=0.5, patch_size=32, inplace=False)
            if t == "Crop":
                self.compose_transforms.register(Crop((96, 96, 96), "random", resize=True), probability=0.5)
            if t == "rotation":
                self.compose_transforms.register(rotation, angles=5, order=0, probability=0.5)
            if t == "random_cutout":
                self.compose_transforms.register(random_cutout, patch_size_ratio=0.4, min_size_ratio=0.1,
                                                 on_data=True, probability=1)
            if t == "cutout_with_threshold":
                self.compose_transforms.register(cutout_with_threshold, patch_sizes=[1, 52, 65, 52], probability=1)

    def __call__(self, x):
        return self.compose_transforms(x)
