"""
Classes and function used to process Geotiff images (used for images' prediction).
"""

# Standard imports
import logging
import os
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import rasterio
import rasterio.windows
from numpy.lib.stride_tricks import as_strided


# Useful functions definitions
def cut_with_overlap_strided(img_data, patch_size_large, patch_size_small):
    """
    Patches of size patch_size_large are extracted while only the center patch of size patch_size_small does not overlap
    For example if the model takes in input patches of 512 and returns patches of 388, this function is suited as it
    will cut the image with overlapped patches of 'patch_size_large'=512 (but without duplicating the overlaps) so that
    the output 'patch_size_small'=388 will cover the whole original image.
    /!\ The patch size has to be a full multiple of the area to extract from otherwise it won't work !

    :param img_data: input image with shape (nr. of bands, height, width)
    :param patch_size_large: tuple with final patch size in form (patch_size_y, patch_size_x)
    :param patch_size_small: tuple with patch size that does not overlap in form (patch_size_y, patch_size_x)
    :return: numpy array with 5 dimensions, shape (#patches_y, #patches_x, bands, patch_size_y, patch_size_x)
    """
    c, h, w = img_data.shape
    assert h >= patch_size_large[0] and w >= patch_size_large[1]
    if (h - (patch_size_large[0] - patch_size_small[0])) % patch_size_small[0] != 0 or (
            w - (patch_size_large[1] - patch_size_small[1])) % patch_size_small[1] != 0:
        logging.warning("The patch size is not a full multiple of the area to extract from")
    i_max = 1 + (h - patch_size_large[0]) // patch_size_small[0]
    j_max = 1 + (w - patch_size_large[1]) // patch_size_small[1]
    X = as_strided(img_data,
                   shape=(i_max, j_max, img_data.shape[0], patch_size_large[0], patch_size_large[1]),
                   strides=(img_data.strides[1] * patch_size_small[0],
                            img_data.strides[2] * patch_size_small[1],
                            img_data.strides[0],
                            img_data.strides[1],
                            img_data.strides[2]))
    return X


def join(cut_image_data):
    """
    Joins an image that was cut with the previous function 'cut_with_overlap_strided' back to the original image.
    """
    cells_x = cut_image_data.shape[1]
    cells_y = cut_image_data.shape[0]
    size_x = cut_image_data.shape[4]
    size_y = cut_image_data.shape[3]
    bands = cut_image_data.shape[2]
    return cut_image_data.transpose(1, 4, 0, 3, 2). \
        reshape(cells_x * size_x, size_y, cells_y, bands). \
        transpose(1, 2, 0, 3). \
        reshape(cells_y * size_y, cells_x * size_x, bands). \
        transpose(2, 0, 1)


def transform_to_categorial_encoding_band(data):
    """
    Convert the Scene Classification Map band (SCM, stored in 12th band) into 12 one-hot encoding bands. Otherwise, all
    input bands are kept.

    /!\ NB: As numpy dims start at 0, B1 is actually data[0, :, :], so data[11, :, :] corresponds to the SCM band.

    :param data: Input data array, data.shape is (number_of_bands, height, width).
    :return: updated data array (number_of_bands + 12, height, width).
    """
    assert(data.shape[0] == 13)  # This function is not flexible and requires 13 bands in the input normalized tiff
    classification_classes = 12
    data_new = np.zeros((data.shape[0] + classification_classes, data.shape[1], data.shape[2]))
    data_new[:data.shape[0]] = data[:data.shape[0]]
    for classification_class in range(classification_classes):
        data_new[data.shape[0] + classification_class][data[11] == classification_class] = 1
    return data_new


# Class definitions
class TifProcessor(ABC):
    """
    Handles the IO to read and write tif files in a way defined by this class. It uses a TifDataProcessor class instance
    to do the processing of the data.
    Example: Expose data that a pixel wise operation can be applied, or expose data in windows of a specific shape to
    optimize memory access and to be able to extract patches of a specific form.
    """

    def __init__(self, input_tif, output_tif, tif_data_processor, output_init_value=-1):
        self.input_tif = input_tif
        self.output_tif = output_tif
        self.tif_dataprocessor = tif_data_processor
        self.rio_input = rasterio.open(self.input_tif, 'r')
        self.meta_output = dict(self.rio_input.meta)
        self.meta_output.update(tif_data_processor.get_meta_changes())
        self.rio_output = rasterio.open(self.output_tif, 'w+', **self.meta_output)

    @abstractmethod
    def process(self):
        """
        Calls the tif_data_processor.process_data_chunk(data) function (multiple times) to process the whole image.
        :return:
        """
        ...


class TifDataProcessor(ABC):
    """
    Processes the data it obtains from the TifProcessor class.
    Note that it is possible that a specific TiffTransformer is not compatible with a specific TifProcessor
    """

    @abstractmethod
    def process_data_chunk(self, input_data):
        """
        The implementation of the data transformation operation on one chunk (to be repeated for all chunks of an image)
        The function works on c x h x w images and returns them in the same shape.


        :param input_data: data chunk with dimension color x height x width
        :return: processed block in form c x h x w
        """
        ...

    @abstractmethod
    def get_meta_changes(self):
        """
        The new metadata that the produced tif should contain (only changes have to be specified)
        Returns metadata updates for the processed data

        :return: for example {'dtype':'uint8', 'count':1} if the output tif has only one band and is in uint8
        """
        ...


class PixelBasedTifProcessor(TifProcessor):
    """
    Can be used to efficiently apply operations that work on all pixels individually (no dependencies on surrounding pixels)
    """

    def __init__(self, input_tif, output_tif, pixelbased_tiftransformer):
        TifProcessor.__init__(self, input_tif, output_tif, pixelbased_tiftransformer)
        self.bands_to_read = [i for i in range(1, self.rio_input.meta['count'] + 1)]
        self.processed = False

    def reset(self):
        """
        Reinitializes the class to allow calling the function "process" a second time and to overwrite previous results.
        :return:
        """
        self.__init__(self.input_tif, self.output_tif, self.tif_dataprocessor)

    def process(self):
        if self.processed:
            logging.error("Image was already processed. Run reset() if reprocessing is wished")
            return

        for ij, window in self.rio_input.block_windows():
            input_data = self.rio_input.read(self.bands_to_read, window=window)
            self.rio_output.write(self.tif_dataprocessor.process_data_chunk(input_data), window=window)

        self.rio_input.close()
        self.rio_output.close()
        self.processed = True


class ToBinaryPixelBasedDataProcessor(TifDataProcessor):
    """
    DataProcessor to map "probability" values that are between 0 and 1 to binary values.
    A cut-off value decides whether they are mapped to 0 or 255
    """

    def __init__(self, cutoff_value=0.9):
        """
        :param cutoff_value: Values below the cutoff_value will be set to 0, those that are larger or equal to 255
        """
        self.cutoff_value = cutoff_value

    def process_data_chunk(self, input_data):
        input_data[input_data >= self.cutoff_value] = 255
        input_data[input_data < self.cutoff_value] = 0
        return np.uint8(input_data)

    def get_meta_changes(self):
        return {'dtype': 'uint8'}


class OverlapWindowTifProcessor(TifProcessor):
    def __init__(self, input_tif, output_tif, overlapwindow_tifdataprocessor, read_window_shape, write_window_shape):
        TifProcessor.__init__(self, input_tif, output_tif, overlapwindow_tifdataprocessor)
        self.bands_to_read = [i for i in range(1, self.rio_input.meta['count'] + 1)]
        self.read_window_shape = read_window_shape
        self.write_window_shape = write_window_shape
        self.processed = False

    def set_bands_to_use(self, band_list):
        """
        Allows to process only a subset of bands. Has to be called before using process()
        :param band_list: List of integers with band numbers (first band is number 1)
                          for example [1, 2, 3] to process only the first three bands
        """
        self.bands_to_read = band_list

    def process(self):
        if self.processed:
            logging.error("Image was already processed. Run reset() if reprocessing is wished")
            return

        h, w = self.rio_input.shape
        logging.info("Begin to process image of shape {}".format(self.rio_input.shape))
        i_max = 1 + (h - self.read_window_shape[0]) // self.write_window_shape[0]
        j_max = 1 + (w - self.read_window_shape[1]) // self.write_window_shape[1]
        i_write_offset = (self.read_window_shape[0] - self.write_window_shape[0]) // 2
        j_write_offset = (self.read_window_shape[1] - self.write_window_shape[1]) // 2
        for i in range(i_max):
            for j in range(j_max):
                logging.info("processing next window {:03d}/{:03d}".format(i, i_max))
                read_window = rasterio.windows.Window(j * self.write_window_shape[1], i * self.write_window_shape[0],
                                                      self.read_window_shape[1], self.read_window_shape[0])
                write_window = rasterio.windows.Window(j_write_offset + j * self.write_window_shape[1],
                                                       i_write_offset + i * self.write_window_shape[0],
                                                       self.write_window_shape[1], self.write_window_shape[0])
                logging.info("read_window: {}, write_window: {}".format(read_window, write_window))
                data = self.rio_input.read(self.bands_to_read, window=read_window)
                logging.info("read data junk size: {}".format(data.shape))
                self.rio_output.write(self.tif_dataprocessor.process_data_chunk(data), window=write_window)
        self.rio_input.close()
        self.rio_output.close()
        self.processed = True


class OverlapPredictionDataProcessor(TifDataProcessor):
    """
    Processes the data with model.predict() for models with a difference in input shape and output shape.
    Assumes that the model uses the channel as last dimension of the input data.
    """

    def __init__(self, model, input_patch_shape='auto', output_patch_shape='auto'):
        self.model = model
        if input_patch_shape == 'auto':
            input_shape = model.layers[0].input_shape[0]
            self.input_patch_shape = (input_shape[1], input_shape[2])
        else:
            self.input_patch_shape = input_patch_shape
        if output_patch_shape == 'auto':
            output_shape = model.layers[-1].output_shape
            self.output_patch_shape = (output_shape[1], output_shape[2])
        else:
            self.output_patch_shape = output_patch_shape

    def process_data_chunk(self, data):
        data = transform_to_categorial_encoding_band(data)
        cut_data = cut_with_overlap_strided(data, patch_size_large=self.input_patch_shape,
                                            patch_size_small=self.output_patch_shape)
        shape = (cut_data.shape[0], cut_data.shape[1], 1, *self.output_patch_shape)
        cut_data = cut_data.reshape(cut_data.shape[0] * cut_data.shape[1], *cut_data.shape[2:]).transpose(0, 2, 3, 1)
        logging.info("cut data shape: {}".format(cut_data.shape))
        y = self.model.predict(x=cut_data)
        logging.info("y shape: {}".format(y.shape))
        return join(y.transpose(0, 3, 1, 2).reshape(shape))

    def get_meta_changes(self):
        return {'count': 1}


