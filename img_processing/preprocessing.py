# -*- coding: utf-8 -*-
"""
Module that defines common transformations that can be applied when the dataset
is loaded.
"""

# Imports
import numpy as np
from skimage import transform as sk_tf
from scipy.ndimage import convolve


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr):
        return self.std * (arr - np.mean(arr))/(np.std(arr) + self.eps) + self.mean


class Crop(object):
    """Crop the given n-dimensional array either at a random location or centered"""

    def __init__(self, shape, type="center", resize=False, keep_dim=False):
        """:param
        shape: tuple or list of int
            The shape of the patch to crop
        type: 'center' or 'random'
            Whether the crop will be centered or at a random location
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

    def __str__(self):
        return f"Padding(shape={self.shape})"


class Binarize(object):

    def __init__(self, one_values=None, threshold=None):
        """ Initialize the instance.
            Parameters
            ----------
            one_values: list
                the value of the input to be set to 1 in the output
            threshold: float
                threshold above which the values of the input will be set to 1 in the output
            """
        self.threshold = threshold
        self.one_values = one_values
        if self.one_values is None and self.threshold is None:
            self.threshold = 0
        
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
        bin_arr = np.zeros_like(arr)
        if self.one_values is not None:
            bin_arr[np.isin(arr, self.one_values)] = 1
        if self.threshold is not None:
            bin_arr[arr > self.threshold] = 1
        return bin_arr

    def __str__(self):
        string = "Binarization("
        if self.threshold is not None:
            string += f"threshold={self.threshold}"
        if self.one_values is not None:
            string += f" one values={self.one_values}"
        string += ")"
        return string


class GaussianSmoothing(object):

    def __init__(self, sigma, size, axes):
        """ Initialize the instance.
            Parameters
            ----------
            size : the size of the Gaussian kernel
            sigma :
        """
        self.size = size
        self.radius = size // 2 + 1
        self.sigma = sigma
        self.axes = axes
    
    def make_gaussian_kernel(self, n_dim):
        """ Create a gaussian kernel with size and sigma
        """
        half_size = self.size // 2
        rng = np.arange(-half_size, half_size + 1, 1)
        x, y, z = np.meshgrid(rng, rng, rng)
        kernel = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * self.sigma ** 2))
        if n_dim > kernel.ndim:
            kernel = np.expand_dims(kernel, axis=[i for i in range(n_dim-kernel.ndim)])
        return kernel.astype(np.float32)
    
    def __call__(self, arr):
        """ Convolution of an array with a gaussian kernel
        """
        kernel = self.make_gaussian_kernel(arr.ndim)
        gauss_arr = convolve(arr, kernel, mode='constant', cval=0.0, origin=0)
        return gauss_arr
    
    def __str__(self):
        return f"GaussianSmoothing(sigma={self.sigma}, size={self.size})"
