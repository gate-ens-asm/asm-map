"""
Script handling the prediction computation (predict ASM locations on images using a selected model)
"""

# Standard imports
import os
import sys
import logging

# Local imports
sys.path.append('.')
import src.utils.log_management as log
import src.utils.predict_utils as pu


def predict_dataset(model_path: str, imgs_folder_path: str, output_folder_path: str, binary_threshold: float):
    """
    Run prediction on all images from the input dataset's folder. This is a necessary step to compute metrics.

    :param str model_path: Path to the model that will be used for prediction. Model's format expected is "xxx.h5".
    :param str imgs_folder_path: Path to the folder that contains all the prepared boxes tiff files to predict.
    :param str output_folder_path: Path to the folder aiming to contain the output predictions. It will be automatically
                                   created, as well as the needed sub-folders, if not already created.
    :param float binary_threshold: Threshold value in [0:1] to convert prediction from probability to binary values.
                                   NB: Recommended value is 0.5
    """
    try:
        assert os.path.isfile(model_path) and os.path.isdir(imgs_folder_path)  # Obviously
        assert len(os.listdir(imgs_folder_path)) > 0  # At least one image to process
        assert 0 <= binary_threshold <= 1.0  # Threshold value must be in [0:1]
    except AssertionError as ae:
        print("ERROR: Wrong input arguments. The model file must exist, images' folder must contain at least 1 tiff, \
               and the Threshold value must stand within [0 : 1]:\n", ae)
        return -1

    # Logging setup
    log.setup_logging_for_prediction(output_folder_path, 'predict-dataset')

    # Physically construct output tree storage folders and get paths to its sub-folders
    proba_prediction_folder_path, binary_prediction_folder_path = pu.create_tree_prediction_folder(output_folder_path)

    # Loop on the desired dataset's boxes (database.GridLabeledRegion objects)
    dataset_imgs_list = [img.split('.')[0] for img in os.listdir(imgs_folder_path)]
    for i, grid_labeled_region in enumerate(dataset_imgs_list):

        # Predict current image if not already done
        if not os.path.isfile(os.path.join(binary_prediction_folder_path,
                                           '{}_binary.tiff'.format(grid_labeled_region))):
            logging.info('{} / {} - Box {}'.format(i + 1, len(dataset_imgs_list), str(grid_labeled_region)))
            raster_box_path = os.path.join(imgs_folder_path, '{}.tiff'.format(grid_labeled_region))
            logging.info('Make prediction...')
            out = pu.predict_box(model_path, raster_box_path, proba_prediction_folder_path, binary_threshold)
            if out is not None:
                return out
        else:
            logging.info('{} / {} - Box {} was already predicted. Skipping.'.format(i + 1, len(dataset_imgs_list),
                                                                                    grid_labeled_region))
