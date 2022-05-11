"""
Script handling the model's evaluation in terms of performances (metrics computation by comparing prediction to reality)
"""

# Standard imports
import datetime
import logging
import os
import re
import sys

# Local imports
sys.path.append('.')
import src.utils.evaluate_utils as eu
import src.utils.log_management as log

# DEFINITION OF GLOBAL VARIABLE : THE PROBABILITY THRESHOLD VALUES FOR METRICS WHICH NEED DISCRETE T VALUES
# NB: these metrics are currently the biomes metric, the aggregated metric and the confusion matrices.
# NB: expected nb of values is 3 /!\
thresholds_array = [0.2, 0.4, 0.6]


def evaluate_all_metrics(pred_folder_path: str, gt_folder_path: str, eval_folder_path: str, gt_asm_shapefile: str,
                         aoi_shapefile_path: str = ''):
    """
    Computes performance metrics for given predictions and corresponding ground truth. Metrics computed are :

    - 1) Global metrics VS threshold value (Precision, Recall, F(0.5) score and MCC) [1 GRAPH]
    - 2) Recall VS threshold per mine's size classification (4 distinct classes) [1 GRAPH]
    - 3) Global metrics per eco-region, for 3 threshold values defined in global var "thresholds_array" [1 TABLE]
    - 4) Receiver Operating Characteristic (ROC) and Area Under Curve (AUC) [1 GRAPH]
    - 5) Confusion Matrices for 3 threshold values defined in global var "thresholds_array" [3 GRAPHS]

    Note that output results will be saved in an "evaluation" folder next to the input GT and predictions folders. Also,
    a post-processing step is optional : a river mask can be applied to the predictions before to compute the metrics.

    :param str pred_folder_path: Path to the folder that contains the predicted tif images (probability predictions).
    :param str gt_folder_path: Path to the folder where the corresponding ground truth tif images are stored.
                               NB: They should have the same filename as the files stored in the "pred_folder_path".
    :param str eval_folder_path: Path to the folder where the output metrics will be stored.
    :param str gt_asm_shapefile: Path to the shapefile containing the GT ASM shapes, corresponding to above predictions.
    :param str aoi_shapefile_path: Path to the area of interest (i.e. the predicted region perimeter) shapefile.
                                   NB: this param is optional, if not given, all the African biomes are considered.
                                   Ex: `asm-map/data/shapefiles/subtropical_west_africa_region.shp`
    """
    try:
        assert os.path.isdir(pred_folder_path) and os.path.isdir(gt_folder_path) and os.path.isfile(gt_asm_shapefile)
        assert os.path.listdir(pred_folder_path) == os.path.listdir(gt_folder_path)  # Same images contained in both
        assert not os.path.samefile(pred_folder_path, gt_folder_path)  # Otherwise, they will be overwritten
        assert len(thresholds_array) == 3  # Otherwise, the script will not work properly
    except AssertionError as ae:
        logging.error('Wrong input arguments. The paths must exist and contain tifs, and there must be 3 T values:')
        logging.error(ae)
        return -1

    # Paths management
    if not os.path.isdir(eval_folder_path):
        os.makedirs(eval_folder_path)
    metrics_csv_path = eu.create_metrics_csv_file(eval_folder_path)  # Create output CSV with main metrics saved

    # Logging setup
    log.setup_logging_for_evaluation(eval_folder_path, 'all-metrics')

    # Compute metrics
    logging.info("FIRST METRIC : PERFORMANCES ALONG THRESHOLDS\n")
    output_fig_path_m1 = os.path.join(eval_folder_path, "1_performances_vs_thresholds.png")
    if not os.path.exists(output_fig_path_m1):  # Check if already exists
        eu.compute_metrics_vs_thresholds(pred_folder_path, gt_folder_path, output_fig_path_m1, False, False,
                                         metrics_csv_path)
    else:
        logging.info(f'Performances VS thresholds metric already computed and available at {output_fig_path_m1}\n')

    logging.info("SECOND METRIC : RECALLS ALONG MINE SIZE CLASSIFICATION\n")
    output_fig_path_m2 = os.path.join(eval_folder_path, "2_recalls_vs_thresholds_per_size_classification.png")
    if not os.path.exists(output_fig_path_m2):  # Check if already exists
        eu.compute_recall_per_size_vs_thresholds(pred_folder_path, gt_asm_shapefile, output_fig_path_m2)
    else:
        logging.info(f'Recalls along mine-size metric already computed and available at {output_fig_path_m2}\n')

    logging.info("THIRD METRIC : PERFORMANCES PER BIOMES\n")
    output_csv_path_m3 = os.path.join(eval_folder_path, "3_performances_per_biomes.csv")
    if not os.path.exists(output_csv_path_m3):  # Check if already exists
        eu.compute_metrics_per_biomes(pred_folder_path, gt_folder_path, output_csv_path_m3, thresholds_array,
                                      aoi_shapefile_path)
    else:
        logging.info(f'Performances per biomes already computed and available at {output_csv_path_m3}\n')

    logging.info("FOURTH METRIC : ROC CURVE + AUC\n")
    output_fig_path_m4 = os.path.join(eval_folder_path, "4_roc_auc_curve.png")
    if not os.path.exists(output_fig_path_m4):  # Check if already exists
        eu.compute_roc_auc_curve(pred_folder_path, gt_folder_path, output_fig_path_m4, metrics_csv_path)
    else:
        logging.info(f'ROC metric already computed and available at {output_fig_path_m4}\n')

    logging.info("FIFTH METRIC : CONFUSION MATRICES\n")
    output_fig_path_m5_1 = os.path.join(eval_folder_path, f"5_confusion_matrix_T{thresholds_array[0]}.png")
    output_fig_path_m5_2 = os.path.join(eval_folder_path, f"5_confusion_matrix_T{thresholds_array[1]}.png")
    output_fig_path_m5_3 = os.path.join(eval_folder_path, f"5_confusion_matrix_T{thresholds_array[2]}.png")
    if (not os.path.exists(output_fig_path_m5_1) or (not os.path.exists(output_fig_path_m5_2))
            or (not os.path.exists(output_fig_path_m5_3))):  # If all already exist, do not compute it again
        eu.compute_confusion_matrices(pred_folder_path, gt_folder_path, eval_folder_path, thresholds_array)
    else:
        logging.info(f'Confusion Matrices already computed and available at {output_fig_path_m5_1[:-7]}X.png\n')


def evaluate_threshold_metric(pred_folder_path: str, gt_folder_path: str, eval_folder_path: str, no_mcc: bool,
                              no_fscore: bool, output_filename: str):
    """
    Computes global performance metrics VS threshold value (Precision, Recall, F(0.5) score and MCC) [GRAPH] for given
    predictions and corresponding ground truth. Note that output graph will be saved in an "evaluation" folder next to
    the input GT and predictions folders.

    :param str pred_folder_path: Path to the folder that contains the predicted tif images (probability predictions).
    :param str gt_folder_path: Path to the folder where the corresponding ground truth tif images are stored.
                               NB: They should have the same filename as the files stored in the "pred_folder_path".
    :param str eval_folder_path: Path to the folder where the output metrics will be stored.
    :param bool no_mcc: Decide whether to plot MCC metric or not, set to False to plot it.
    :param bool no_fscore: Decide whether to plot MCC metric or not, set to False to plot it.
    :param str output_filename: Filename with extension for the output graph.
    """
    try:
        assert os.path.isdir(pred_folder_path) and os.path.isdir(gt_folder_path)  # Obviously
        assert os.path.listdir(pred_folder_path) == os.path.listdir(gt_folder_path)  # Same images contained in both
        assert not os.path.samefile(pred_folder_path, gt_folder_path)  # Otherwise, they will be overwritten
    except AssertionError as ae:
        logging.error('Wrong input arguments. The paths must exist and contain the same tifs filenames:')
        logging.error(ae)
        return -1

    # Paths management
    if not os.path.isdir(eval_folder_path):
        os.makedirs(eval_folder_path)
    metrics_csv_path = eu.create_metrics_csv_file(eval_folder_path)  # Create output CSV with main metrics saved

    # Logging setup
    log.setup_logging_for_evaluation(eval_folder_path, 'threshold-metric')

    logging.info("FIRST METRIC : PERFORMANCES ALONG THRESHOLDS\n")
    output_fig_path_m1 = os.path.join(eval_folder_path, output_filename)
    if not os.path.exists(output_fig_path_m1):  # Check if already exists
        eu.compute_metrics_vs_thresholds(pred_folder_path, gt_folder_path, output_fig_path_m1, no_mcc, no_fscore,
                                         metrics_csv_path)
    else:
        logging.info(f'Performances VS thresholds metric already computed and available at {output_fig_path_m1}\n')


def evaluate_size_metric(pred_folder_path: str, gt_asm_shapefile: str, eval_folder_path: str):
    """
    Computes Recall VS thresholds per mine's size classification (4 distinct classes + industrial mines class) [GRAPH],
    for given predictions and corresponding ground truth.

    :param str pred_folder_path: Path to the folder that contains the predicted tif images (probability predictions).
    :param str gt_asm_shapefile: Path to the shapefile containing the GT ASM shapes, corresponding to above predictions.
    :param str eval_folder_path: Path to the folder where the output metrics will be stored.
    """
    try:
        assert os.path.isfile(gt_asm_shapefile)  # Obviously
        assert os.path.isdir(pred_folder_path) and len(os.path.listdir(pred_folder_path)) > 0  # Existing and not empty
    except AssertionError as ae:
        logging.error('Wrong input arguments. The paths must exist and contain tifs files:')
        logging.error(ae)
        return -1

    # Paths management and logging setup
    if not os.path.isdir(eval_folder_path):
        os.makedirs(eval_folder_path)
    log.setup_logging_for_evaluation(eval_folder_path, 'size-metric')

    logging.info("SECOND METRIC : RECALLS ALONG MINE SIZE CLASSIFICATION\n")
    output_fig_path_m2 = os.path.join(eval_folder_path, "2_recalls_vs_thresholds_per_size_classification.png")
    if not os.path.exists(output_fig_path_m2):  # Check if already exists
        eu.compute_recall_per_size_vs_thresholds(pred_folder_path, gt_asm_shapefile, output_fig_path_m2)
    else:
        logging.info(f'Recalls along mine-size metric already computed and available at {output_fig_path_m2}\n')


def evaluate_biomes_metric(pred_folder_path: str, gt_folder_path: str, eval_folder_path: str,
                           aoi_shapefile_path: str = ''):
    """
    Computes global metrics per biome [TABLE], for 3 threshold values defined in global variable "thresholds_array"
    and for given predictions and corresponding ground truth.

    :param str pred_folder_path: Path to the folder that contains the predicted tif images (probability predictions).
    :param str gt_folder_path: Path to the folder where the corresponding ground truth tif images are stored.
                               NB: They should have the same filename as the files stored in the "pred_folder_path".
    :param str eval_folder_path: Path to the folder where the output metrics will be stored.
    :param str aoi_shapefile_path: Path to the area of interest (i.e. the predicted region perimeter) shapefile.
                                   NB: this param is optional, if not given, all the African biomes are considered.
                                   Ex: `asm-map/data/shapefiles/subtropical_west_africa_region.shp`
    """
    try:
        assert os.path.isdir(pred_folder_path) and os.path.isdir(gt_folder_path)  # Obviously
        assert os.path.listdir(pred_folder_path) == os.path.listdir(gt_folder_path)  # Same images contained in both
        assert not os.path.samefile(pred_folder_path, gt_folder_path)  # Otherwise, they will be overwritten
    except AssertionError as ae:
        logging.error('Wrong input arguments. The paths must exist and contain the same tifs filenames:')
        logging.error(ae)
        return -1

    # Paths management and logging setup
    if not os.path.isdir(eval_folder_path):
        os.makedirs(eval_folder_path)
    log.setup_logging_for_evaluation(eval_folder_path, 'size-metric')

    logging.info("THIRD METRIC : PERFORMANCES PER BIOMES\n")
    output_csv_path_m3 = os.path.join(eval_folder_path, "3_performances_per_biomes.csv")
    if not os.path.exists(output_csv_path_m3):  # Check if already exists
        eu.compute_metrics_per_biomes(pred_folder_path, gt_folder_path, output_csv_path_m3, thresholds_array,
                                      aoi_shapefile_path)
    else:
        logging.info(f'Performances per biomes already computed and available at {output_csv_path_m3}\n')


def evaluate_roc_metric(pred_folder_path: str, gt_folder_path: str, eval_folder_path: str):
    """
    Computes Receiver Operating Characteristic (ROC) graph and Area Under Curve value (AUC) for given ground truth and
    predictions tiff files.

    :param str pred_folder_path: Path to the folder that contains the predicted tif images (probability predictions).
    :param str gt_folder_path: Path to the folder where the corresponding ground truth tif images are stored.
                               NB: They should have the same filename as the files stored in the "pred_folder_path".
    :param str eval_folder_path: Path to the folder where the output metrics will be stored.
    """
    try:
        assert os.path.isdir(pred_folder_path) and os.path.isdir(gt_folder_path)  # Obviously
        assert os.path.listdir(pred_folder_path) == os.path.listdir(gt_folder_path)  # Same images contained in both
        assert not os.path.samefile(pred_folder_path, gt_folder_path)  # Otherwise, they will be overwritten
    except AssertionError as ae:
        logging.error('Wrong input arguments. The paths must exist and contain the same tifs filenames:')
        logging.error(ae)
        return -1

    # Paths management
    if not os.path.isdir(eval_folder_path):
        os.makedirs(eval_folder_path)
    metrics_csv_path = eu.create_metrics_csv_file(eval_folder_path)  # Create output CSV with main metrics saved

    # Logging setup
    log.setup_logging_for_evaluation(eval_folder_path, 'roc-metric')

    logging.info("FOURTH METRIC : ROC CURVE + AUC\n")
    output_fig_path_m4 = os.path.join(eval_folder_path, "4_roc_auc_curve.png")
    if not os.path.exists(output_fig_path_m4):  # Check if already exists
        eu.compute_roc_auc_curve(pred_folder_path, gt_folder_path, output_fig_path_m4, metrics_csv_path)
    else:
        logging.info(f'ROC metric already computed and available at {output_fig_path_m4}\n')


def evaluate_matrix_metric(pred_folder_path: str, gt_folder_path: str, eval_folder_path: str):
    """
    Computes and plot Confusion Matrices, for 3 threshold values defined in global variable "thresholds_array" and for
    given ground truth and predictions tiff files.

    :param str pred_folder_path: Path to the folder that contains the predicted tif images (probability predictions).
    :param str gt_folder_path: Path to the folder where the corresponding ground truth tif images are stored.
                               NB: They should have the same filename as the files stored in the "pred_folder_path".
    :param str eval_folder_path: Path to the folder where the output metrics will be stored.
    """
    try:
        assert os.path.isdir(pred_folder_path) and os.path.isdir(gt_folder_path)  # Obviously
        assert os.path.listdir(pred_folder_path) == os.path.listdir(gt_folder_path)  # Same images contained in both
        assert not os.path.samefile(pred_folder_path, gt_folder_path)  # Otherwise, they will be overwritten
        assert len(thresholds_array) == 3  # Otherwise, the script will not work properly
    except AssertionError as ae:
        logging.error('Wrong input arguments. The paths must exist and contain tifs, and there must be 3 T values:')
        logging.error(ae)
        return -1

    # Paths management and logging setup
    if not os.path.isdir(eval_folder_path):
        os.makedirs(eval_folder_path)
    log.setup_logging_for_evaluation(eval_folder_path, 'size-metric')

    logging.info("FIFTH METRIC : CONFUSION MATRICES\n")
    output_fig_path_m5_1 = os.path.join(eval_folder_path, f"5_confusion_matrix_T{thresholds_array[0]}.png")
    output_fig_path_m5_2 = os.path.join(eval_folder_path, f"5_confusion_matrix_T{thresholds_array[1]}.png")
    output_fig_path_m5_3 = os.path.join(eval_folder_path, f"5_confusion_matrix_T{thresholds_array[2]}.png")
    if (not os.path.exists(output_fig_path_m5_1) or not os.path.exists(output_fig_path_m5_2)
            or not os.path.exists(output_fig_path_m5_3)):  # If all already exist, do not compute it again
        eu.compute_confusion_matrices(pred_folder_path, gt_folder_path, eval_folder_path, thresholds_array)
    else:
        logging.info(f'Confusion Matrices already computed and available at {output_fig_path_m5_1[:-7]}X.png\n')
