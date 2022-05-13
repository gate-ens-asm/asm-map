"""
Useful functions used in main predict command (images prediction using an already trained model)
"""

# Standard imports
import os
import sys
from pathlib import Path

# Third party imports
import tensorflow as tf

# Local imports
sys.path.append('..')
from src.utils import tif_processing as tp
from src.utils.loss_metrics_utils import focal_loss, f05_m, recall_m, precision_m

os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Paths related functions
def create_tree_prediction_folder(base_folder_path_str: str):
    """
    Create and organize the output base folder path into a tree:
     - "prediction" sub-folder : for visualization of the predictions made on each box. This sub-folder is divided into
       2 sub-folders named "probability" and "binary", each containing one tiff for each image (named accordingly).

    :param str base_folder_path_str: Output folder path (will be created physically if not already exists).
    :returns: the paths (str) to the 'prediction/probability' and 'prediction/binary' sub-folders.
    """
    proba_prediction_folder_path_str = os.path.join(base_folder_path_str, 'prediction', 'probability')
    binary_prediction_folder_path_str = os.path.join(base_folder_path_str, 'prediction', 'binary')

    base_folder_path = Path(base_folder_path_str)
    base_folder_path.mkdir(parents=True, exist_ok=True)
    prediction_probability_path = Path(proba_prediction_folder_path_str)
    prediction_probability_path.mkdir(parents=True, exist_ok=True)
    prediction_binary_path = Path(binary_prediction_folder_path_str)
    prediction_binary_path.mkdir(parents=True, exist_ok=True)

    return proba_prediction_folder_path_str, binary_prediction_folder_path_str


# Prediction related functions
def predict_box(model_path: str, image_path: str, proba_prediction_folder_path: str, binary_threshold: float):
    """
    Predict the input image with the selected model.
    NB: It saves the probability (and then binary) predictions into the previously constructed associated subfolders.

    :param str model_path: Path to the model that will be used for prediction. Model's format expected is "xxx.h5".
    :param str image_path: Path to the current box' preprocessed tiff file.
    :param str proba_prediction_folder_path: Path to output probability prediction folder (assumed as already created).
    :param float binary_threshold: Threshold value in [0:1] to convert prediction from probability to binary values.
    """
    # First, define model's useful parameters
    input_patch_size = 572
    output_patch_size = 388
    window_size = 1449

    try:
        # Second, load keras model
        model = tf.keras.models.load_model(model_path, custom_objects={'f05_metric': f05_m, 'fl': focal_loss(),
                                                                       'recall_m': recall_m, 'precision_m': precision_m})

        # Third, define output useful paths
        image_id = int(os.path.splitext(os.path.basename(image_path))[0])
        bin_prediction_folder_path = os.path.join(os.path.dirname(proba_prediction_folder_path), 'binary')
        proba_prediction_tif_path = os.path.join(proba_prediction_folder_path, '{}.tiff'.format(image_id))
        binary_prediction_tif_path = os.path.join(bin_prediction_folder_path, '{}_binary.tiff'.format(image_id))

        # Fourth, compute the input/output patches sizes
        wins = (window_size - (input_patch_size - output_patch_size)) // output_patch_size
        read_window_shape_w = wins * output_patch_size + (input_patch_size - output_patch_size)
        read_window_shape_h = input_patch_size
        write_window_shape_w = wins * output_patch_size
        write_window_shape_h = output_patch_size

        # Fifth, create probability prediction
        data_processor = tp.OverlapPredictionDataProcessor(model, input_patch_shape='auto', output_patch_shape='auto')
        processor = tp.OverlapWindowTifProcessor(image_path, proba_prediction_tif_path, data_processor,
                                                 read_window_shape=(read_window_shape_h, read_window_shape_w),
                                                 write_window_shape=(write_window_shape_h, write_window_shape_w))
        processor.process()

        # Sixth, create binary prediction
        data_processor = tp.ToBinaryPixelBasedDataProcessor(cutoff_value=binary_threshold)
        processor = tp.PixelBasedTifProcessor(proba_prediction_tif_path, binary_prediction_tif_path, data_processor)
        processor.process()
    except Exception as e:
        logging.error('A problem occurred during the current prediction: ', e)
        return -1
