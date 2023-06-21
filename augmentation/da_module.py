from augmentation import Transformer
from augmentation.intensity import add_blur, add_noise
from augmentation.spatial import flip, cutout
from preprocessing.transforms import Crop


class DA_Module(object):

    def __init__(self, transforms=("flip", "add_blur", "add_noise", "cutout", "crop")):
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

    def __call__(self, x):
        return self.compose_transforms(x)
