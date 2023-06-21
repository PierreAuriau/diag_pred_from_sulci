# -*- coding: utf-8 -*-
"""
Module that defines common transformations that can be applied when the dataset
is loaded.
"""

# Imports
import collections
import logging
import numpy as np
from scipy.ndimage import rotate, affine_transform, convolve
from skimage import transform as sk_tf
import torch
import torch.nn.functional as F

logger = logging.getLogger("")

class Scaler(object):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, data):
        return self.scale * data


class LabelMapping(object):

    def __init__(self, **mappings):
        self.mappings = mappings

    def __call__(self, label):
        if isinstance(label, list) or isinstance(label, np.ndarray):
            l_to_return = []
            for l in label:
                l_to_return.append(self.__call__(l))
            return l_to_return
        if label in self.mappings:
            return self.mappings[label]
        else:
            return label


class HardNormalization(object):
    def __init__(self, min=-1.0, max=1.0, eps=1e-8):
        self.min = min
        self.max = max
        self.eps = eps

    def __call__(self, arr):
        min_arr = np.min(arr)
        max_arr = np.max(arr)
        if np.abs(min_arr - max_arr) < self.eps:
            return np.zeros_like(arr)
        return ((self.max - self.min) * arr + (self.min * max_arr - self.max * min_arr)) / (max_arr - min_arr)


class RandomFlip(object):
    def __init__(self, vflip=False, hflip=True, dflip=True, proba=0.5):
        self.vflip = vflip
        self.hflip = hflip
        self.dflip = dflip
        self.prob = proba

    def __call__(self, arr):
        if self.vflip and np.random.rand() < self.prob:
            arr = np.flip(arr, axis=[2])
        if self.hflip and np.random.rand() < self.prob:
            arr = np.flip(arr, axis=[1])
        if self.dflip and np.random.rand() < self.prob:
            arr = np.flip(arr, axis=[0])
        return arr.copy()


class RandomPatchInversion(object):
    def __init__(self, patch_size=10, data_threshold=0):
        self.data_threshold = data_threshold
        self.patch_size = patch_size

    def __call__(self, arr, label=None):
        assert isinstance(arr, np.ndarray)
        if label is None:
            label = int(np.random.rand() < 0.5)
        assert label in [0, 1], "Unexpected label"
        if label == 1:
            # Selects 2 random non-overlapping patch
            mask = (arr > self.data_threshold)
            # Get a first random patch inside the mask
            patch1 = self.get_random_patch(mask)
            # Get a second one outside the first patch and inside the mask
            mask[patch1] = False
            patch2 = self.get_random_patch(mask)
            arr = arr.copy()
            data_patch1 = arr[patch1].copy()
            arr[patch1] = arr[patch2]
            arr[patch2] = data_patch1
            print(patch1, patch2)
        return arr, label

    def get_random_patch(self, mask):
        # Warning: we assume the mask is convex
        possible_indices = mask.nonzero(as_tuple=True)
        if len(possible_indices[0]) == 0:
            raise ValueError("Empty mask")
        index = np.random.randint(len(possible_indices[0]))
        point = [min(ind[index], mask.shape[i] - self.patch_size) for i, ind in enumerate(possible_indices)]
        patch = tuple([slice(p, p + self.patch_size) for p in point])
        return patch


class Random90_3DRot(object):
    """Applies a rotation in {0, 90, 180, 270} in each direction and returns a label k in [0..23]"""

    def __init__(self, authorized_rot=None, axes=None):
        if authorized_rot is not None:
            assert set(authorized_rot) <= {0, 90, 180, 270}
        self.authorized_rot = list(authorized_rot or [0, 90, 180, 270])
        self.nb_rots = len(self.authorized_rot)
        self.num_classes = self.nb_rots * 3 * 2  # 3 axes, 2 directions or the nb of faces in a cube
        self.rot_to_k = {0: 0, 90: 1, 180: 2, 270: 3}

        # The 'front' is the axes (1, 2) here. It is arbitrary.
        self.authorized_axes = [(1, 2), (1, 3), (2, 3)]
        self.cube_face_to_back = [(2, (1, 3)), (1, (3, 2)), (1, (3, 1))]
        self.cube_face_to_front = [(0, (1, 2)), (1, (2, 3)), (1, (1, 3))]

        self.cube_face = None
        if axes is not None:
            assert axes in self.authorized_axes, "Axes must be in {}".format(self.authorized_axes)
            self.num_classes = self.nb_rots
            self.cube_face = 2 * self.authorized_axes.index(axes)

        # Small tests to confirm that everything is ok (i.e the transformation T: label -> T(I) is injective
        # for any image I)
        self.test_unicity()

    def __call__(self, arr, label=None):
        assert len(arr.shape) == 4 and isinstance(arr, np.ndarray)
        # Chose a label
        if label is None:
            label = np.random.randint(0, self.num_classes)
        assert label in np.arange(self.num_classes), "Unexpected label"
        # If the cube's face is already predefined, use it
        if self.cube_face is not None:
            cube_face = self.cube_face
            angle = self.authorized_rot[label]
        else:
            # Get the associated angle and cube's face
            angle_index, cube_face = np.unravel_index(label, (self.nb_rots, 6))
            angle = self.authorized_rot[angle_index]

        # From the cube's face, deduce the axis and direction
        (direction, face_axes) = (cube_face % 2, self.authorized_axes[cube_face // 2])

        # Put the selected face to front or back (front is the axes (0, 1) in 3D, (1, 2) in 4D with the channel)
        if direction == 0:
            (k, axes) = self.cube_face_to_front[cube_face // 2]
            arr = np.rot90(arr, k=k, axis=axes)

        elif direction == 1:
            (k, axes) = self.cube_face_to_back[cube_face // 2]
            arr = np.rot90(arr, k=k, axis=axes)

        # Rotate of the chosen angle in the direction selected
        arr = np.rot90(arr, k=self.rot_to_k[angle], axis=face_axes)
        return arr, label

    def test_unicity(self):
        m = torch.arange(8).reshape((1, 2, 2, 2))
        list_permutation = []
        for k in range(self.num_classes):
            rotated_m, label = self(m, label=k)
            for m_0 in list_permutation:
                if rotated_m == m_0:
                    raise ValueError("Several labels map to the same configuration")
            list_permutation.append(rotated_m)
        return list_permutation


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, arr):
        return self.std * (arr - np.mean(arr)) / (np.std(arr) + self.eps) + self.mean


class Standardize(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        if np.isscalar(self.std):
            if self.std == 0:
                self.std = 1
        if isinstance(self.std, np.ndarray):
            self.std[self.std == 0.0] = 1

    def __call__(self, arr):
        return (arr - self.mean) / self.std


class Crop(object):
    """Crop the given n-dimensional array either at a random location or centered"""

    def __init__(self, shape, type="center", resize=False, keep_dim=False):
        """:param
        shape: tuple or list of int
            The shape of the patch to crop
        type: 'center' or 'random'
            Wheter the crop will be centered or at a random location
        resize: bool, default False
            If True, resize the cropped patch to the inital dim. If False, depends on keep_dim
        keep_dim: bool, default False
            if True and resize==False, put a constant value around the patch cropped. If resize==True, does nothing
        """
        assert type in ["center", "random"]
        self.shape = shape
        self.copping_type = type
        self.resize = resize
        self.keep_dim = keep_dim

    def __call__(self, arr):
        assert isinstance(arr, np.ndarray)
        assert type(self.shape) == int or len(self.shape) == len(arr.shape), "Shape of array {} does not match {}". \
            format(arr.shape, self.shape)

        img_shape = np.array(arr.shape)
        if type(self.shape) == int:
            size = [self.shape for _ in range(len(self.shape))]
        else:
            size = np.copy(self.shape)
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.copping_type == "center":
                delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
            elif self.copping_type == "random":
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(delta_before, delta_before + size[ndim]))
        if self.resize:
            # resize the image to the input shape
            return sk_tf.resize(arr[tuple(indexes)], img_shape, preserve_range=True)

        if self.keep_dim:
            mask = np.zeros(img_shape, dtype=np.bool)
            mask[tuple(indexes)] = True
            arr_copy = arr.copy()
            arr_copy[~mask] = 0
            return arr_copy

        return arr[tuple(indexes)]


class Resize(object):

    def __init__(self, output_shape, **kwargs):
        self.kwargs = kwargs
        self.output_shape = output_shape

    def __call__(self, arr):
        return sk_tf.resize(arr, self.output_shape, **self.kwargs)


class Rescale(object):

    def __init__(self, scale, **kwargs):
        self.kwargs = kwargs
        self.scale = scale

    def __call__(self, arr):
        return sk_tf.rescale(arr, self.scale, **self.kwargs)


class GaussianNoise:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, arr):
        return arr + self.std * torch.randn_like(arr)


class RandomAffineTransform3d:
    def __init__(self, angles, translate):
        ## angles == list of int or tuple indicating the range of degrees to select from in each direction
        ## translate == tuple of maximum absolute fraction translation shift in each direction
        if type(angles) in [int, float]:
            angles = [[-angles, angles] for _ in range(3)]
        elif type(angles) == list and len(angles) == 3:
            for i in range(3):
                if type(angles[i]) in [int, float]:
                    angles[i] = [-angles[i], angles[i]]
        else:
            raise ValueError("Unkown angles type: {}".format(type(angles)))
        self.angles = angles
        if type(translate) in [float, int]:
            translate = (translate, translate, translate)
        assert len(translate) == 3
        self.translate = translate

    def __call__(self, arr):
        assert len(arr.shape) == 4 and isinstance(arr, np.ndarray)  # == (C, H, W, D)

        arr_shape = np.array(arr.shape)
        angles = [np.deg2rad(np.random.random() * (angle_max - angle_min) + angle_min)
                  for (angle_min, angle_max) in self.angles]
        alpha, beta, gamma = angles[0], angles[1], angles[2]
        rot_x = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
        rot_y = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        rot_z = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        R = np.matmul(np.matmul(rot_z, rot_y), rot_x)
        middle_point = (np.asarray(arr_shape[1:]) - 1) / 2
        offset = middle_point - np.dot(middle_point, R)

        translation = [np.round(np.random.random() * (2 * arr_shape[i + 1] * t) - arr_shape[i + 1] * t)
                       for i, t in enumerate(self.translate)]
        out = np.zeros(arr_shape, dtype=arr.dtype)
        for c in range(arr_shape[0]):
            affine_transform(arr[c], R.T, offset=offset + translation, output=out[c], mode='nearest')

        return out


class Rotation(object):
    # TODO : convert it to handle torch tensors
    def __init__(self, angle, axes=(1, 2), reshape=True, **kwargs):
        self.angle = angle
        self.axes = axes
        self.reshape = reshape
        self.rotate_kwargs = kwargs

    def __call__(self, arr):
        return rotate(arr, self.angle, axes=self.axes, reshape=self.reshape, **self.rotate_kwargs)


class RandomRotation(object):
    # TODO: convert it to handle torch tensors
    """ nd generalisation of https://pytorch.org/docs/stable/torchvision/transforms.html section RandomRotation"""

    def __init__(self, angles, axes=(0, 2), reshape=True, probability=None, with_channels=True, **kwargs):
        if type(angles) in [int, float]:
            self.angles = [-angles, angles]
        elif type(angles) == list and len(angles) == 2 and angles[0] < angles[1]:
            self.angles = angles
        else:
            raise ValueError("Unkown angles type: {}".format(type(angles)))
        if isinstance(axes, tuple):
            self.axes = [axes]
        elif isinstance(axes, list):
            self.axes = axes
        elif axes is None:
            print('Warning: rotation plane will be determined randomly')
            self.axes = [tuple(np.random.choice(3, 2, replace=False))]
        else:
            raise ValueError(f"Unknown axes type : {type(axes)}")
        self.reshape = reshape
        self.probability = probability
        self.with_channels = with_channels
        self.rotate_kwargs = kwargs

    def __call__(self, arr):
        if self.probability is not None and np.random.random() > self.probability:
            return arr
        if self.with_channels:
            data = []
            for _arr in arr:
                data.append(self._apply_random_rotation(_arr))
            return np.asarray(data)
        else:
            return self._apply_random_rotation(arr)

    def _apply_random_rotation(self, arr):
        angle = np.float16(np.random.uniform(self.angles[0], self.angles[1]))
        for axes in self.axes:
            arr = rotate(arr, angle, axes=axes, reshape=self.reshape, **self.rotate_kwargs)
        return arr

    def __str__(self):
        string = f"Random Rotation : angles={self.angles}, axes={self.axes}"
        if self.probability is not None:
            string += f"| probability={self.probability}"
        return string


class Padding(object):
    """ A class to pad an image.
    """

    def __init__(self, shape, **kwargs):
        """ Initialize the instance.

        Parameters
        ----------
        shape: list of int
            the desired shape.
        **kwargs: kwargs given to torch.nn.functional.pad()
        """
        self.shape = shape
        self.kwargs = kwargs

    def __call__(self, arr):
        """ Fill an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array.

        Returns
        -------
        fill_arr: np.array
            the padded array.
        """
        if len(arr.shape) >= len(self.shape):
            return self._apply_padding(arr)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, arr):
        """ See Padding.__call__().
        """
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append([half_shape_i, half_shape_i])
            else:
                padding.append([half_shape_i, half_shape_i + 1])
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append([0, 0])
        fill_arr = np.pad(arr, padding, **self.kwargs)
        return fill_arr


class Downsample(object):
    """ A class to downsample an array.
    """

    def __init__(self, scale, with_channels=True):
        """ Initialize the instance.

        Parameters
        ----------
        scale: int
            the downsampling scale factor in all directions.
        with_channels: bool, default True
            if set expect the array to contain the channels in first dimension.
        """
        self.scale = scale
        self.with_channels = with_channels

    def __call__(self, arr):
        """ Downsample an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array

        Returns
        -------
        down_arr: np.array
            the downsampled array.
        """
        if self.with_channels:
            data = []
            for _arr in arr:
                data.append(self._apply_downsample(_arr))
            return np.asarray(data)
        else:
            return self._apply_downsample(arr)

    def _apply_downsample(self, arr):
        """ See Downsample.__call__().
        """
        slices = []
        for cnt, orig_i in enumerate(arr.shape):
            if cnt == 3:
                break
            slices.append(slice(0, orig_i, self.scale))
        down_arr = arr[tuple(slices)]

        return down_arr


class Binarize(object):

    def __init__(self, one_values=None, threshold=None, with_channels=True):
        """ Initialize the instance.
            Parameters
            ----------
            one_values: list
                the value of the input to be set to 1 in the output
            threshold: float
                threshold above which the values of the input will be set to 1 in the output
            with_channels: bool, default False
                if set expect the array to contain the channels in first dimension.
                if set, one_values and threshold can be different for each channels
            """
        self.threshold = threshold
        self.one_values = one_values
        if self.one_values is None and self.threshold is None:
            self.threshold = 0
        self.with_channels = with_channels

    def __call__(self, arr):
        """ Binarize an array according to one values.
            Parameters
            ----------
            arr: np.array
                an input array
            Returns
            -------
            bin_arr: np.array
                the binarize array.
       """
        if self.with_channels:
            data = []
            for ch, _arr in enumerate(arr):
                data.append(self._apply_binarize(_arr, self.one_values, self.threshold))
            return np.asarray(data)
        else:
            return self._apply_binarize(arr, self.one_values, self.threshold)

    @staticmethod
    def _apply_binarize(arr, one_values, threshold):
        """ See Binarize.__call__().
        """
        bin_arr = np.zeros_like(arr)
        if one_values is not None:
            bin_arr[np.isin(arr, one_values)] = 1
        if threshold is not None:
            bin_arr[arr > threshold] = 1
        return bin_arr

    def __str__(self):
        string = "Binarization :"
        if self.threshold is not None:
            string += f" threshold={self.threshold}"
        if self.one_values is not None:
            string += f" one values={self.one_values}"
        return string


class GaussianConvolution(object):

    def __init__(self, sigma, size, random=False, with_channels=True):
        """ Initialize the instance.
            Parameters
            ----------
            size : the size of the Gaussian kernel
            sigma : the standard deviation of the Gaussian probabilty distribution
                    in random mode, it should be a tuple (sigma_min, sigma_max)
            random : if True, sigma is pulled from a uniform distribution between sigma_min and sigma_max
        """
        self.size = int(size)
        if self.size % 2 == 0:
            self.size += 1
        self.random = random
        if self.random:
            assert len(sigma) == 2, print("In random mode, sigma should be a tuple (sigma_min, sigma_max)")
        self.sigma = sigma
        self.with_channels = with_channels

    def make_gaussian_kernel(self, sigma):
        """ Create a gaussian kernel with size and sigma
        """
        half_size = self.size // 2
        rng = np.arange(-half_size, half_size + 1, 1)
        x, y, z = np.meshgrid(rng, rng, rng)
        kernel = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
        return kernel.astype(np.float16)

    def __call__(self, arr):
        """ Convolution of an array with a gaussian kernel
        """
        if self.random:
            sigma = np.float16(np.random.uniform(self.sigma[0], self.sigma[1]))
        else:
            sigma = self.sigma
        kernel = self.make_gaussian_kernel(sigma=sigma)
        if self.with_channels:
            data = []
            for ch, _arr in enumerate(arr):
                data.append(convolve(_arr, kernel, mode='constant', cval=0.0, origin=0))
            gauss_arr = np.asarray(data)
        else:
            gauss_arr = convolve(arr, kernel, mode='constant', cval=0.0, origin=0)
        return gauss_arr

    def __str__(self):
        string = f"Gaussian Convolution : kernel size={self.size}"
        if self.random:
            string += f", sigma=[{self.sigma[0]}, {self.sigma[1]}]"
        else:
            string += f", sigma={self.sigma}"
        return string
