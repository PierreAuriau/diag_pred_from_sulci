# -*- coding: utf-8 -*-

"""
Common functions to transform image.
Code: https://github.com/fepegar/torchio
"""

# Import
import numpy as np
from scipy.ndimage import map_coordinates, rotate
import logging

logger = logging.getLogger()

class Rotation(object):

    def __init__(self, angles, axes=(0, 2), reshape=False, with_channels=True, **kwargs):
        if isinstance(angles, (int, float)):
            self.angles = [-angles, angles]
        elif isinstance(angles, (list, tuple)):
            assert (len(angles) == 2 and angles[0] < angles[1]), print(f"Wrong angles format {angles}")
            self.angles = angles
        else:
            raise ValueError("Unkown angles type: {}".format(type(angles)))
        if isinstance(axes, tuple):
            self.axes = [axes]
        elif isinstance(axes, list):
            self.axes = axes
        else:
            logger.warning('Rotations: rotation plane will be determined randomly')
            self.axes = [tuple(np.random.choice(3, 2, replace=False))]
        self.reshape = reshape
        self.with_channels = with_channels
        self.rotate_kwargs = kwargs

    def __call__(self, arr):
        if self.with_channels:
            data = []
            for _arr in arr:
                data.append(self._apply_random_rotation(_arr))
            return np.asarray(data)
        else:
            return self._apply_random_rotation(arr)

    def _apply_random_rotation(self, arr):
        angles = [np.float16(np.random.uniform(self.angles[0], self.angles[1]))
                  for _ in range(len(self.axes))]
        for ax, angle in zip(self.axes, angles):
            arr = rotate(arr, angle, axes=ax, reshape=self.reshape, **self.rotate_kwargs)
        return arr

    def __str__(self):
        return f"Rotation(angles={self.angles}, axes={self.axes})"


class Cutout(object):
    def __init__(self, patch_size, random_size=False, localization="random", with_channels=True, **kwargs):
        self.patch_size = patch_size
        self.random_size = random_size
        self.with_channels = with_channels
        if localization in ["random", "on_data"] or isinstance(localization, (tuple, list)):
            self.localization = localization
        else:
            logger.warning("Cutout : localization is set to random")
            self.localization = "random"
        self.min_size = kwargs.get("min_size", 0)
        self.value = kwargs.get("value", 0)
        self.image_shape = kwargs.get("image_shape", None)
        if self.image_shape is not None:
            self.patch_size = self._set_patch_size(self.patch_size,
                                                   self.image_shape)
            self.min_size = self._set_patch_size(self.min_size,
                                                 self.image_shape)

    def __call__(self, arr):
        if self.image_shape is None:
            arr_shape = arr.shape[1:] if self.with_channels else arr.shape
            self.patch_size = self._set_patch_size(self.patch_size,
                                                   arr_shape)
            self.min_size = self._set_patch_size(self.min_size,
                                                 arr_shape)

        if self.with_channels:
            data = []
            for _arr in arr:
                data.append(self._apply_cutout(_arr))
            return np.asarray(data)
        else:
            return self._apply_cutout(arr)

    def _apply_cutout(self, arr):
        image_shape = arr.shape
        if self.localization == "on_data":
            nonzero_voxels = np.nonzero(arr)
            index = np.random.randint(0, len(nonzero_voxels[0]))
            localization = np.array([nonzero_voxels[i][index] for i in range(len(nonzero_voxels))])
        elif isinstance(self.localization, (tuple, list)):
            assert len(self.localization) == len(image_shape), f"Cutout : wrong localization shape"
            localization = self.localization
        else:
            localization = None
        indexes = []
        for ndim, shape in enumerate(image_shape):
            if self.random_size:
                size = np.random.randint(self.min_size[ndim], self.patch_size[ndim])
            else:
                size = self.patch_size[ndim]
            if localization is not None:
                delta_before = max(localization[ndim] - size // 2, 0)
            else:
                delta_before = np.random.randint(0, shape - size + 1)
            indexes.append(slice(delta_before, delta_before + size))
        arr[tuple(indexes)] = self.value
        return arr

    @staticmethod
    def _set_patch_size(patch_size, image_shape):
        if isinstance(patch_size, int):
            size = [patch_size
                    for _ in range(len(image_shape))]
        elif isinstance(patch_size, float):
            size = [int(patch_size*s) for s in image_shape]
        else:
            size = patch_size
        assert len(size) == len(image_shape), "Incorrect patch dimension."
        for ndim in range(len(image_shape)):
            if size[ndim] > image_shape[ndim] or size[ndim] < 0:
                size[ndim] = image_shape[ndim]
        return size

    def __str__(self):
        return f"Cutout(patch_size={self.patch_size}, random_size={self.random_size}, " \
               f"localization={self.localization})"
