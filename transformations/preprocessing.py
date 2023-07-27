# -*- coding: utf-8 -*-
"""
Module that defines common transformations that can be applied when the dataset
is loaded.
"""

# Imports
import numpy as np
from skimage import transform as sk_tf


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

