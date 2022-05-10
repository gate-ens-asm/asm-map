"""
Useful functions used in main evaluate commands (metrics computation)
"""

# Standard imports
import csv
import datetime
import logging
import os
import re
import sys

# Third party imports
import geoalchemy2.shape
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio.mask
import seaborn as sn
import shapely.geometry
from matplotlib.transforms import Bbox
from shapely.ops import transform
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, fbeta_score
from sklearn.metrics import roc_curve, auc, confusion_matrix


# 1) Main metrics functions
def compute_metrics_vs_thresholds(pred_folder_path: str, gt_folder_path: str, output_fig_path: str, no_mcc: bool,
                                  no_fscore: bool, metrics_csv_path: str):
    """
    Computes global performances' metrics (Precision, Recall, F(0.5) score, Matthews Correlation Coefficient) for
    varying probability threshold in [0 : 1].

    :param str pred_folder_path: Path to the folder that contains the prediction boxes with probability values in [0:1].
    :param str gt_folder_path: Path to the folder that contains the GT boxes, corresponding to above predictions.
    :param str output_fig_path: Output figure to be saved path, expected extension is PNG or JPG.
    :param bool no_mcc: Decide whether to plot MCC metric or not, set to False to plot it.
    :param bool no_fscore: Decide whether to plot MCC metric or not, set to False to plot it.
    :param str metrics_csv_path: Path to the already created CSV in which are saved the key metrics, for comparison.
    """
    try:
        assert (os.path.isdir(pred_folder_path) and os.path.isdir(gt_folder_path) and os.path.isfile(metrics_csv_path))
        assert (len(os.listdir(pred_folder_path)) > 0)
        assert (len(os.listdir(gt_folder_path)) == len(os.listdir(pred_folder_path)))
    except AssertionError as e:
        logging.error('Input folders/files do not exist and/or do not contain necessary files:\n{}'.format(e))
        logging.info(f'Prediction folder path : {pred_folder_path}\nGT folder path : {gt_folder_path}\n Metrics CSV \
                     path : {metrics_csv_path}')
        return -1

    thresholds_m1 = np.arange(0, 1.023, 0.033)  # 30 thresholds in [0:1]
    logging.info('Nb of thresholds values considered : {}'.format(len(thresholds_m1)))

    # Create numpy arrays with predictions (proba) and ground truth values for all of the dataset's boxes
    gt_array, proba_pred_array = create_global_dataset_np_arrays(pred_folder_path, gt_folder_path)  # TODO

    # Loop on thresholds and compute metrics
    precision_scores = [compute_precision_per_threshold(gt_array, proba_pred_array, t) for t in thresholds_m1]
    recall_scores = [compute_recall_per_threshold(gt_array, proba_pred_array, t) for t in thresholds_m1]
    if not no_fscore:
        f05_scores = [compute_fbeta_per_threshold(gt_array, proba_pred_array, t, 0.5) for t in thresholds_m1]
    if not no_mcc:
        mcc_scores = [compute_mcc_per_threshold(gt_array, proba_pred_array, t) for t in thresholds_m1]

    # Retrieve best results and save output graph
    if (not no_fscore) or (not no_mcc):
        logging.info("BEST METRICS OBTAINED :")
    if not no_fscore:
        save_best_metric_to_csv(metrics_csv_path, 'F_score', f05_scores, thresholds_m1)
    if not no_mcc:
        save_best_metric_to_csv(metrics_csv_path, 'MCC', mcc_scores, thresholds_m1)
    save_key_P_R_metrics_to_csv(metrics_csv_path, precision_scores, recall_scores, thresholds_m1)

    plt.figure()
    plt.plot(thresholds_m1, precision_scores, "-", c='cornflowerblue', label="Precision")
    plt.plot(thresholds_m1, recall_scores, "-", c='limegreen', label="Recall")
    if not no_fscore:
        plt.plot(thresholds_m1, f05_scores, "--", c='orange', label="F(0.5) Score")
    if not no_mcc:
        plt.plot(thresholds_m1, mcc_scores, "--", c='hotpink', label="MCC")
    plt.xlabel("Probability Threshold"), plt.xlim([0., 1.]), plt.ylim([0., 1.])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.5)
    plt.grid(b=True, which="both", axis="both", color='gray', linestyle='-', linewidth=1)
    plt.title("Model's performances evolution along Probability Threshold", pad='10.0', fontsize=15)
    plt.savefig(output_fig_path, bbox_inches=Bbox([[-1, -1], [9, 5]]))
    logging.info("Performances along thresholds metric successfully computed and saved.\n")


def compute_recall_per_size_vs_thresholds(pred_folder_path: str, gt_asm_shapefile: str, output_fig_path: str):
    """
    Computes global recall for varying probability threshold in [0 : 1], by regrouping mines within size classes.
    Mine-size classes are the following (created from the stats computed on our dataset to form homogeneous classes):
      - Micro ASM from 0.1 to 1 hectare;
      - Small ASM from 1 to 2.5 hectares;
      - Medium ASM from 2.5 to 7 hectares;
      - Macro ASM from 7 to 5000 hectares;
      - Large-scale mines (industrial mines), discriminated from the others thanks to the label value = 1.

    :param str pred_folder_path: Path to the folder that contains the prediction boxes with probability values in [0:1].
    :param str gt_asm_shapefile: Path to the shapefile containing the GT ASM shapes, corresponding to above predictions.
    :param str output_fig_path: Output figure to be saved path, expected extension is PNG or JPG.
    """
    # Init needed variables (such as mine's size classification and probability threshold range)
    size_conditions = [(0, 10000), (10000, 25000), (25000, 70000), (70000, 50000000)]
    thresholds_m2 = np.arange(0.02, 1.02, 0.02)
    asm_recalls_dict = dict()
    logging.info(f"Nb of thresholds values considered : {len(thresholds_m2)}")

    # Get GT shapes list from shapefile
    ground_truth_shapes = get_asm_shapes_from_gt_shapefile(gt_asm_shapefile)  # TODO

    # Compute projected area and retrieve predictions values for each GT mine shapes
    asm_dicts_list = create_dict_with_areas_and_preds_for_gt_shapes(ground_truth_shapes, pred_folder_path, 0)
    indus_dicts_list = create_dict_with_areas_and_preds_for_gt_shapes(ground_truth_shapes, pred_folder_path, 1)
    logging.info(f"Nb of ASM mines in studied dataset : {len(asm_dicts_list)}")
    logging.info(f"Nb of industrial mines in studied dataset : {len(indus_dicts_list)}")

    # Compute recall metric for each mine size class ASM, then for industrial mines separately
    for c in size_conditions:
        asm_recalls_dict[f'size{c[0]}_{c[1]}'] = [compute_recall_for_mine_size_per_threshold(asm_dicts_list, t, c)
                                                  for t in thresholds_m2]
    indus_recalls_list = [compute_recall_for_mine_size_per_threshold(indus_dicts_list, t, (0, 50000000))
                          for t in thresholds_m2]

    # Visualization and save
    plt.figure()
    plt.plot(thresholds_m2, asm_recalls_dict['size0_10000'], "--", c='darkorange', label="Recall (S in 0-1 hectare)")
    plt.plot(thresholds_m2, asm_recalls_dict['size10000_25000'], "--", c='yellow', label="Recall (S in 1-2.5 hectares)")
    plt.plot(thresholds_m2, asm_recalls_dict['size25000_70000'], "--", c='chartreuse',
             label="Recall (S in 2.5-7 hectares)")
    plt.plot(thresholds_m2, asm_recalls_dict['size70000_50000000'], "--", c='forestgreen',
             label="Recall (S in 7-2000 hectares)")
    plt.plot(thresholds_m2, indus_recalls_list, "--", c='darkslategrey', label="Recall for industrial mines")
    # plt.axvline(x=0.1), plt.axvline(x=0.2), plt.axvline(x=0.6)
    plt.ylim([0, 101])
    plt.title("Recall per mine's size classification vs Probability Threshold")
    plt.grid(b=True, which="both", axis="both", color='gray', linestyle='-', linewidth=1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("Probability Threshold"), plt.ylabel("Recall")
    plt.savefig(output_fig_path, bbox_inches=Bbox([[-1, -1], [9, 5]]))
    logging.info("Mine-sized classes recalls along thresholds metric successfully computed and saved.\n")


def compute_metrics_per_biomes(pred_folder_path: str, gt_folder_path: str, output_csv_path: str, thresholds,
                               aoi_shapefile_path: str = ''):
    """
    Computes global performances metrics (Precision, Recall, F(0.5) score, Matthews Correlation Coefficient) for
    varying probability thresholds defined in global variable "thresholds_array", by regrouping the data along their
    biome (= eco-region) affiliation.

    :param str pred_folder_path: Path to the folder that contains the prediction boxes with probability values in [0:1].
    :param str gt_folder_path: Path to the folder that contains the GT boxes, corresponding to above predictions.
    :param str output_csv_path: Output figure to be saved path, expected extension is CSV.
    :param list thresholds: Array with probability threshold values (floats) in [0 : 1].
    :param str aoi_shapefile_path: Path to the area of interest (i.e. the predicted region perimeter) shapefile.
                                   NB: this param is optional, if not given, all the African biomes are considered.
                                   Ex: `asm-map/data/shapefiles/subtropical_west_africa_region.shp`
    """
    # Manage shapefiles and dataframes, query boxes and init useful variables
    am_map_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # TODO
    biomes_shp = os.path.join(am_map_folder, "data/shapefiles/africa_biomes_wwf_final.shp")  # TODO
    biomes_df = gpd.read_file(biomes_shp)
    # aoi_region_shp = os.path.join(am_map_folder, "data/region/real_west_africa_tropical_rainforest_region.shp")
    if aoi_shapefile_path != '' and os.path.exists(aoi_shapefile_path):  # Intersect df to keep only biomes in the aoi
        aoi_region_df = gpd.read_file(aoi_shapefile_path)
        aoi_biomes_df = gpd.clip(biomes_df, aoi_region_df)
    else:  # If aoi not specified or incorrect, keep all of the African biomes
        logging.warning("\nNo Area Of Interest (aoi) taken into account, all Africa considered as the predicted area.")
        logging.warning("It means that the biome's areas and cover ratios may not be representative of what you want.")
        aoi_biomes_df = biomes_df

    # Retrieve dataset's boxes' information
    all_boxes_dataset = create_dict_with_boxes_infos(pred_folder_path)  # TODO

    metrics_all_biomes = []  # Initiate output metrics list
    areas_biomes, areas_biomes_percentage = compute_areas_biomes(aoi_biomes_df)  # Get biomes areas

    for biome_index in aoi_biomes_df.index:  # Loop on the biomes in aoi

        # Init biome's values
        nb_boxes = 0
        biome_gt = []
        biome_pred = []
        logging.info(f"\nBIOME : {aoi_biomes_df.BIOME_NUM[biome_index]} - {aoi_biomes_df.BIOME_NAME[biome_index]}")

        for dataset_box in all_boxes_dataset:  # Loop on boxes and retrieve paths
            gt_path = os.path.join(gt_folder_path, str(dataset_box.grid_labeled_regions_id) + '.tiff')  # TODO
            pred_path = os.path.join(pred_folder_path, str(dataset_box.grid_labeled_regions_id) + '.tiff')  # TODO
            center_of_box = geoalchemy2.shape.to_shape(dataset_box.geom).centroid  # TODO

            # Add to biome's array if the box center is into the biome
            if center_of_box.within(aoi_biomes_df.geometry[biome_index]):
                nb_boxes += 1
                np_gt, np_pred_proba = get_np_arrays(gt_path, pred_path)  # TODO
                biome_gt.extend(np_gt), biome_pred.extend(np_pred_proba)

        # Compute metrics upon this biome, then append results to output global list with all the biomes
        if nb_boxes != 0:
            logging.info(f'Nb of boxes in biome : {nb_boxes}')
            logging.info(f'Nb of pixels in biome : {len(biome_gt) / 1000000} M')
            biome_precisions = [compute_precision_per_threshold(biome_gt, biome_pred, t) for t in thresholds]
            biome_recalls = [compute_recall_per_threshold(biome_gt, biome_pred, t) for t in thresholds]
            biome_f05s = [compute_fbeta_per_threshold(biome_gt, biome_pred, t, 0.5) for t in thresholds]
            biome_mccs = [compute_mcc_per_threshold(biome_gt, biome_pred, t) for t in thresholds]

            metrics_biome = [
                aoi_biomes_df.BIOME_NUM[biome_index],
                aoi_biomes_df.BIOME_NAME[biome_index],
                float('%f' % (areas_biomes[biome_index] / 1000000)),  # km² with no decimal
                areas_biomes_percentage[biome_index],  # % with 1 decimal
            ]
            for i in range(len(thresholds)):
                metrics_biome.append(float('%.2f' % (biome_precisions[i])))
                metrics_biome.append(float('%.2f' % (biome_recalls[i])))
                metrics_biome.append(float('%.2f' % (biome_f05s[i])))
                metrics_biome.append(float('%.2f' % (biome_mccs[i])))
            metrics_all_biomes.append(metrics_biome)

    # Final DataFrame's creation and save it in output CSV file
    df_columns = ['BIOME NUM', 'BIOME NAME', 'BIOME SURFACE (km²)', 'BIOME COVERAGE (%)']
    for t in thresholds:
        df_columns.append(f'PRECISION (T={t})')
        df_columns.append(f'RECALL (T={t})')
        df_columns.append(f'F(0.5) SCORE (T={t})')
        df_columns.append(f'MCC (T={t})')
    metrics_biomes_df = pd.DataFrame(metrics_all_biomes, columns=df_columns)
    metrics_biomes_df.to_csv(output_csv_path)
    logging.info("Performances per biomes metric successfully computed and saved.\n")


def compute_roc_auc_curve(pred_folder_path: str, gt_folder_path: str, output_fig_path: str, metrics_csv_path: str):
    """
    Computes Receiver Operating Characteristic (ROC) graph and Area Under Curve value (AUC) for given ground truth and
    predictions tiff files.

    :param str pred_folder_path: Path to the folder that contains the prediction boxes with probability values in [0:1].
    :param str gt_folder_path: Path to the folder that contains the GT boxes, corresponding to above predictions.
    :param str output_csv_path: Output filepath to be saved to, expected extension is CSV.
    :param str metrics_csv_path: Path to the already created CSV in which are saved the key metrics, for comparison.
    """
    try:
        assert (os.path.isdir(pred_folder_path) and os.path.isdir(gt_folder_path) and os.path.isfile(metrics_csv_path))
        assert (len(os.listdir(pred_folder_path)) > 0)
        assert (len(os.listdir(gt_folder_path)) == len(os.listdir(pred_folder_path)))
    except AssertionError as ae:
        logging.error('Input folders/files do not exist and/or do not contain necessary files:\n{}'.format(ae))
        logging.info(f'Prediction folder path : {pred_folder_path}\nGT folder path : {gt_folder_path}\n Metrics CSV \
                     path : {metrics_csv_path}')
        return -1

    # Create numpy arrays with predictions (proba) and ground truth values for all of the dataset's boxes
    gt_array, proba_pred_array = create_global_dataset_np_arrays(pred_folder_path, gt_folder_path)

    # Compute ROC and AUC with sklearn.metrics
    fpr, tpr, thresholds = roc_curve(gt_array, proba_pred_array)
    auc_value = auc(fpr, tpr)

    # Save AUC metric to main CSV metrics file
    append_metric_row_to_csv_metrics_file(
        metrics_csv_path,
        [7, 'AUC', float('%.3f' % auc_value), datetime.datetime.now().strftime('%d-%m-%Y')]
    )

    # Create graph visualisation + save figure to output path
    plt.figure()
    plt.plot(fpr, tpr, "-", c='cornflowerblue', lw=1.5, label='ROC curve (AUC = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')  # Plot reference x = y line
    plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0]), plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", borderaxespad=0.5)
    plt.grid(b=True, which="both", axis="both", color='gray', linestyle='-', linewidth=1)
    plt.title('Receiver operating characteristic (ROC)', pad='10.0', fontsize=15)
    plt.savefig(output_fig_path, bbox_inches=Bbox([[-1, -1], [7, 5]]))
    logging.info("ROC metric successfully computed and saved.\n")


def compute_confusion_matrices(pred_folder_path: str, gt_folder_path: str, eval_folder_path: str, thresholds):
    """
    Computes and plot Confusion Matrices, for given threshold values and for given ground truth and predictions tiff
    files. The script checks if the files already exist before overwriting them.
    NB : CM = [[ TN , FP],
               [ FN , TP]]

    :param str pred_folder_path: Path to the folder that contains the prediction boxes with probability values in [0:1].
    :param str gt_folder_path: Path to the folder that contains the GT boxes, corresponding to above predictions.
    :param str eval_folder_path: Path to the output folder in which the figures will be saved.
    :param list thresholds: Array containing the probability threshold values (floats) in [0 : 1].
    """
    # Create numpy arrays with predictions (proba) and ground truth values for all of the dataset's boxes
    gt_array, proba_pred_array = create_global_dataset_np_arrays(pred_folder_path, gt_folder_path)

    # Loop on Probability threshold values and compute each corresponding Confusion Matrix with sklearn.metrics
    cm_array = [compute_cm_per_threshold(gt_array, proba_pred_array, t) for t in thresholds]

    # Prepare general visualization settings
    labels = ["True Negatives", "False Positives", "False Negatives", "True Positives"]
    categories = ['Not Mine', 'Mine']
    fig_size = (5, 5)

    # Create each Confusion Matrix visualization and save it to given output path
    for cm, threshold in zip(cm_array, thresholds):
        fig_title = f"Confusion matrix (for Probability Threshold = {threshold})"  # Set figure title
        output_fig_path_m5 = os.path.join(eval_folder_path, f"5_confusion_matrix_T{threshold}.png")
        if not os.path.exists(output_fig_path_m5):  # Check if file already exists
            for i in range(len(cm)):  # Convert nb of pixels to hectares (approx via 1 pix = 100 m²)
                cm[i] = cm[i] / 100
            # Create dataframe from cm nd-array for plotting
            df_cm = pd.DataFrame(cm, index=[i for i in ["True Negatives", "True Positives"]],
                                 columns=[j for j in ["False Negatives", "False Positives"]])
            plot_confusion_matrix(df_cm.to_numpy(), output_fig_path_m5, group_names=labels, categories=categories,
                                  figsize=fig_size, cmap='Blues', title=fig_title)
            logging.info(f'Confusion Matrix (T={threshold}) successfully computed, available at {output_fig_path_m5}\n')
        else:
            logging.info(f'Confusion Matrix (T={threshold}) already computed and available at {output_fig_path_m5}\n')


# 2) Key metrics saved to CSV related functions
def create_metrics_csv_file(eval_folder_path: str):
    """
    Creates output CSV file that aims to contain all the main key metrics, if it does not already exist.

    :param eval_folder_path: Path to the output evaluation folder which will contain the metrics graphs and tables.
    :return: CSV path (str), whose filename will be "main_metrics.csv"
    """
    assert(os.path.isdir(eval_folder_path))
    csv_metrics_file = os.path.join(eval_folder_path, 'main_metrics.csv')
    if os.path.exists(csv_metrics_file):
        logging.info('CSV file with the main metrics already exists at :\n{}'.format(csv_metrics_file))
        return csv_metrics_file
    else:
        with open(csv_metrics_file, 'w', encoding='UTF8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            header = ['id', 'metric', 'value', 'computation_date']
            writer.writerow(header)
        logging.info('CSV file successfully created at :\n{}'.format(csv_metrics_file))
        return csv_metrics_file


def check_if_main_metric_already_saved(csv_metrics_file: str, metric_id: int):
    """
    Check if a specific metric is already saved into the metrics CSV file (check performed on the metric_id).

    :param str csv_metrics_file: Path to the already created CSV file in which are saved the key metrics.
    :param int metric_id: ID of the key metric to be checked (positive int).
    :return: boolean (True if metric is already saved in CSV, False otherwise)
    """
    assert(os.path.isfile(csv_metrics_file))
    with open(csv_metrics_file, 'r', encoding='UTF8') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        metrics_ids = [int(row[0]) for row in reader]
        if metric_id in metrics_ids:
            return True
        else:
            return False


def append_metric_row_to_csv_metrics_file(csv_metrics_file: str, metric_row):
    """
    Add key metric into the CSV file that stores the main performance metrics used for comparison.

    :param str csv_metrics_file: Path to the already created CSV file in which are saved the key metrics.
    :param list metric_row: Array containing the metrics' 4 elements to be added into the CSV. Its format must be:
                            [metric_index (int), metric_name (str), metric_value (float), computation_date ('%d-%m-%Y')]
    """
    try:
        assert (os.path.isfile(csv_metrics_file) and len(metric_row) == 4)
    except AssertionError as e:
        logging.error('Wrong input arguments for append_metric_row_to_csv_metrics_file func:\n{}'.format(e))
        logging.info(f'CSV filepath : {csv_metrics_file}\nMetric row : {metric_row}\n')
        return -1

    if check_if_main_metric_already_saved(csv_metrics_file, int(metric_row[0])):
        logging.warning('This metric was already saved into the main metrics CSV : {}. Aborting.'.format(metric_row))
        return -1

    with open(csv_metrics_file, 'a', encoding='UTF8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(metric_row)
        logging.info(f'Successfully added {metric_row[0]} - {metric_row[1]} to the key metrics CSV file.')


def save_best_metric_to_csv(csv_metrics_file: str, metric_name: str, metric_list, thresholds_list):
    """
    Read max metric value and its corresponding threshold, then save it to the key metrics CSV. (Expect Fscore or MCC)

    :param str csv_metrics_file: Path to the already created CSV file in which are saved the key metrics.
    :param str metric_name: Name of the metric considerd, either Fscore or MCC metric.
    :param list metric_list: List of the metrics' values (floats) computed for each Threshold value.
    :param list thresholds_list: List of Threshold (T) values used to compute the metric (same length as metric_list).
    """
    try:
        assert (os.path.isfile(csv_metrics_file) and len(metric_list) == len(thresholds_list))
    except AssertionError as e:
        logging.error('Wrong input arguments for save_best_metric_to_csv func:\n{}'.format(e))
        logging.info(f'CSV filepath : {csv_metrics_file}\nMetric name : {metric_name}\nMetric list : \
                     {metric_list}\nThresholds list : {thresholds_list}\n')
        return -1

    try:
        max_metric = max(metric_list)
        T_for_max_metric = thresholds_list[metric_list.index(max_metric)]

        if re.search('f.*?score', metric_name.lower()):
            row_max = [0, 'F_max', max_metric, datetime.datetime.now().strftime('%d-%m-%Y')]
            row_T_max = [1, 'T(F_max)', T_for_max_metric, datetime.datetime.now().strftime('%d-%m-%Y')]
            logging.info(f"Max F(0.5) value : {max_metric}")
            logging.info(f"Threshold value for F(0.5) max : {T_for_max_metric}")
        elif re.search('mcc', metric_name.lower()):
            row_max = [2, 'MCC_max', max_metric, datetime.datetime.now().strftime('%d-%m-%Y')]
            row_T_max = [3, 'T(MCC_max)', T_for_max_metric, datetime.datetime.now().strftime('%d-%m-%Y')]
            logging.info(f"Max MCC value : {max_metric}")
            logging.info(f"Threshold value for MCC max : {T_for_max_metric}")
        else:
            logging.error('The metric name was not recognized (neither Fscore nor MCC): {}\n'.format(metric_name))
            return -1

        append_metric_row_to_csv_metrics_file(csv_metrics_file, row_max)
        append_metric_row_to_csv_metrics_file(csv_metrics_file, row_T_max)
    except Exception as e:
        logging.error('Could not compute and/or save the max {} key metric :\n{}'.format(metric_name, e))
        logging.info(f'CSV filepath : {csv_metrics_file}\nMetric name : {metric_name}\nMetric list : \
                     {metric_list}\nThresholds list : {thresholds_list}\n')
        return -1


def save_key_P_R_metrics_to_csv(csv_metrics_file: str, precision_list, recall_list, thresholds_list):
    """
    Read key metrics from the Precisions and Recalls computed, then save them to the key metrics CSV. Key metrics are:
    - Which Threshold value for Precision = Recall (location of the curves intersection) ;
    - Which Precision for Recall = 20% ;
    - Which Precision for Recall = 40% ;
    NB: This func only makes sense for metrics computed on a significant nb of thresholds (e.g. step = 0.01).

    :param csv_metrics_file: Path to the already created CSV file in which are saved the key metrics (str).
    :param precision_list: List of the Precision values computed for each Threshold value.
    :param recall_list: List of the Recall values computed for each Threshold value.
    :param thresholds_list: List of Threshold (T) values used to compute the metric (same length as both metric_lists).
    """
    try:
        assert (os.path.isfile(csv_metrics_file))
        assert (len(precision_list) == len(recall_list) and len(recall_list) == len(thresholds_list))
    except AssertionError as e:
        logging.error('Wrong input arguments for save_key_P_R_metrics_to_csv func:\n{}'.format(e))
        logging.info(f'CSV filepath : {csv_metrics_file}\nPrecision list : {precision_list}\nRecall list : \
                      {recall_list}\nThresholds list : {thresholds_list}\n')
        return -1

    # Find T value corresponding to Precision = Recall (curves intersection location)
    try:
        index_intersect = next(precision_list.index(precision) for precision, recall in zip(precision_list, recall_list)
                               if precision >= recall)
        T_intersect = thresholds_list[index_intersect]
        logging.info(f"Threshold value for Precision/Recall intersection : {T_intersect}")
        row_intersect = [4, 'T(P_cross_R)', T_intersect, datetime.datetime.now().strftime('%d-%m-%Y')]
        append_metric_row_to_csv_metrics_file(csv_metrics_file, row_intersect)
    except Exception as e:
        logging.error(f'Key metric T(P_cross_R) could not be computed and/or saved : {e}')

    # Find Precision score corresponding to Recall = 20%
    try:
        index_recall_02 = next(recall_list.index(recall) for recall in recall_list if recall <= 0.2)
        P_recall_02 = precision_list[index_recall_02]
        logging.info(f"Precision value for Recall = 20% : {P_recall_02}")
        row_P_for_R02 = [5, 'P(R_0.2)', P_recall_02, datetime.datetime.now().strftime('%d-%m-%Y')]
        append_metric_row_to_csv_metrics_file(csv_metrics_file, row_P_for_R02)
    except Exception as e:
        logging.error(f'Key metric P(R_0.2) could not be computed and/or saved : {e}')

    # Find Precision score corresponding to Recall = 40%
    try:
        index_recall_04 = next(recall_list.index(recall) for recall in recall_list if recall <= 0.4)
        P_recall_04 = precision_list[index_recall_04]
        logging.info(f"Precision value for Recall = 40% : {P_recall_04}")
        row_P_for_R04 = [6, 'P(R_0.4)', P_recall_04, datetime.datetime.now().strftime('%d-%m-%Y')]
        append_metric_row_to_csv_metrics_file(csv_metrics_file, row_P_for_R04)
    except Exception as e:
        logging.error(f'Key metric P(R_0.4) could not be computed and/or saved : {e}')


# 3) Metrics computation related functions
def create_mask(data, values_to_keep):
    """
    Creates a mask (same shape as data), True for each value that is in the list "values_to_keep", False elsewhere.

    :param data: The input data to use as basis for the mask
    :param values_to_keep: An array with the values (floats) to set to true in output mask.
    :return: mask with shape of data
    """
    mask = np.zeros_like(data, dtype='bool')  # Create initial mask with False everywhere
    for value in values_to_keep:  # Loop on the input values to be kept and set them to True in the mask
        mask[data == value] = True
    return mask


def get_np_arrays(ground_truth_tif_path: str, prediction_tif_path: str):
    """
    Retrieve and prepare GT and proba predictions into masked flatten arrays, prepared to easily compute metrics.

    :param str ground_truth_tif_path: Path to a ground truth tif file
    :param str prediction_tif_path: Path to a prediction (in probabilities) tif file (which corresponds to GT file)
    :return: np_ground_truth = GT masked binary array (np array(1, nb_pixels_predicted))
    :return: np_pred_proba = Pred masked probability array (np array(1, nb_pixels_predicted))
    """
    # Retrieve and read GT + Pred probability content
    rio_data_ground_truth = rasterio.open(ground_truth_tif_path)
    rio_data_pred = rasterio.open(prediction_tif_path)
    np_ground_truth = rio_data_ground_truth.read(1)
    np_pred_proba = rio_data_pred.read(1)
    prediction_mask = ~ create_mask(np_pred_proba, [-1])

    # Retrieve only prediction zone within the box (exclude -1 values) + Flatten
    np_pred_proba = np_pred_proba[prediction_mask]
    np_ground_truth = np_ground_truth[prediction_mask]

    return np_ground_truth, np_pred_proba


def create_global_dataset_np_arrays(pred_folder_path: str, gt_folder_path: str):
    """
    Retrieve and prepare GT and proba predictions iteratively for all boxes contained in entry folder paths, with a
    possible post-processing step (river masking) into masked flatten arrays. The 2 output arrays contain the
    concatenated content for all boxes and are prepared to easily compute metrics.

    :param str pred_folder_path: Path to the folder that contains the prediction boxes with probability values in [0:1].
    :param str gt_folder_path: Path to the folder that contains the GT boxes, corresponding to above predictions. They
                               should have the same name as the files stored in the "pred_folder_path", ie their ids.
    :return: gt_array = GT masked binary array (np array(1, nb_pixels_predicted_in_all_boxes))
    :return: proba_pred_array = Pred masked probability array (np array(1, nb_pixels_predicted_in_all_boxes))
    """
    proba_pred_array = []  # Init dataset's global prediction array
    gt_array = []  # Init dataset's global ground truth array

    # Loop on boxes in dataset to retrieve proba prediction array and GT array, then add them to global arrays
    for box_filename in os.listdir(pred_folder_path):
        gt_box_path = os.path.join(gt_folder_path, box_filename)
        pred_box_path = os.path.join(pred_folder_path, box_filename)
        np_box_gt, np_box_pred_proba = get_np_arrays(gt_box_path, pred_box_path)
        proba_pred_array.extend(np_box_pred_proba), gt_array.extend(np_box_gt)

    logging.info(f"Nb of pixels processed : {float('%.2f' % (len(gt_array) / 1000000))} M")
    logging.info(f"Content of GT array : {np.unique(gt_array)}")
    logging.info(f"Content of Pred array : {np.unique(proba_pred_array)}")

    return gt_array, proba_pred_array


def apply_binary_threshold_to_prediction(np_pred_proba, threshold: float = 0.5):
    """
    Copy array, then apply probability threshold in [0:1] to given prediction array to obtain output binary array.

    :param nparray np_pred_proba: Numpy array that contains probability values in [0:1] for prediction pixels.
    :param float threshold: Value of the probability threshold, that classifies predictions, in [0:1].
    :return: Numpy array with the same shape than input numpy array, with values in {0, 1}
    """
    np_pred_binary = np.asarray(np_pred_proba).copy()
    np_pred_binary[np_pred_binary >= threshold] = 1
    np_pred_binary[np_pred_binary < threshold] = 0
    return np.uint8(np_pred_binary)


def compute_precision_per_threshold(gt_array, pred_proba_array, threshold: float = 0.5):
    """
    Computes Precision metric in [0 : 1]  by comparing ground truth to prediction.

    :param gt_array: Ground Truth masked binary array (np array(1, nb_pixels_predicted))
    :param pred_proba_array: Prediction masked probability array (np array(1, nb_pixels_predicted))
    :param threshold: Probability threshold value in [0:1], used to convert entry proba prediction to binary (float)
    :return: precision score computed by sklearn.metrics (float with 2 decimals).
    """
    pred_bin_array = apply_binary_threshold_to_prediction(pred_proba_array, threshold)
    precision = precision_score(gt_array, pred_bin_array, average='binary', zero_division=1)
    return float('%.2f' % precision)  # float in [0 : 1] with only 2 decimals


def compute_recall_per_threshold(gt_array, pred_proba_array, threshold: float = 0.5):
    """
    Computes Recall metric in [0 : 1]  by comparing ground truth to prediction.

    :param gt_array: Ground Truth masked binary array (np array(1, nb_pixels_predicted))
    :param pred_proba_array: Prediction masked probability array (np array(1, nb_pixels_predicted))
    :param threshold: Probability threshold value in [0:1], used to convert entry proba prediction to binary (float)
    :return: recall score computed by sklearn.metrics (float with 2 decimals).
    """
    pred_bin_array = apply_binary_threshold_to_prediction(pred_proba_array, threshold)
    recall = recall_score(gt_array, pred_bin_array, average='binary', zero_division=1)
    return float('%.2f' % recall)  # float in [0 : 1] with only 2 decimals


def compute_fbeta_per_threshold(gt_array, pred_proba_array, threshold: float = 0.5, beta: float = 1):
    """
    Computes F(beta) score metric in [0 : 1] by comparing ground truth to prediction.

    :param gt_array: Ground Truth masked binary array (np array(1, nb_pixels_predicted))
    :param pred_proba_array: Prediction masked probability array (np array(1, nb_pixels_predicted))
    :param threshold: Probability threshold value in [0:1], used to convert entry proba prediction to binary (float)
    :param beta: Float value in [0 : +inf] used to put more weight on either precision (ex : 0.5) or recall (ex : 2).
    :return: F(beta) score computed by sklearn.metrics (float with 2 decimals).
    """
    pred_bin_array = apply_binary_threshold_to_prediction(pred_proba_array, threshold)
    f_score = fbeta_score(gt_array, pred_bin_array, beta=beta, average='binary', zero_division=0)
    return float('%.2f' % f_score)  # float in [0 : 1] with only 2 decimals


def compute_mcc_per_threshold(gt_array, pred_proba_array, threshold: float = 0.5):
    """
    Computes Matthews Correlation Coefficient (MCC) metric by comparing ground truth to prediction.

    :param gt_array: Ground Truth masked binary array (np array(1, nb_pixels_predicted))
    :param pred_proba_array: Prediction masked probability array (np array(1, nb_pixels_predicted))
    :param threshold: Probability threshold value in [0:1], used to convert entry proba prediction to binary (float)
    :return: recall score computed by sklearn.metrics (float with 2 decimals).
    """
    pred_bin_array = apply_binary_threshold_to_prediction(pred_proba_array, threshold)
    mcc = matthews_corrcoef(gt_array, pred_bin_array)
    return float('%.2f' % mcc)  # float in [-1 : 1] with only 2 decimals


def compute_cm_per_threshold(gt_array, pred_proba_array, threshold: float = 0.5):
    """
    Computes Confusion Matrix by comparing ground truth to prediction.

    :param gt_array: Ground Truth masked binary array (np array(1, nb_pixels_predicted))
    :param pred_proba_array: Prediction masked probability array (np array(1, nb_pixels_predicted))
    :param threshold: Probability threshold value in [0:1], used to convert entry proba prediction to binary (float)
    :return: confusion matrix scores computed by sklearn.metrics : ndarray of shape (n_classes, n_classes).
    """
    pred_bin_array = apply_binary_threshold_to_prediction(pred_proba_array, threshold)
    cm = confusion_matrix(gt_array, pred_bin_array)
    return cm


def get_projected_area_for_shape(shape):
    """
    Computes projected area of given shape, by transposing to EPSG 3857 - Web Mercator.

    :param shape: Instance of class GridShapes defined in src/utils/database.py (mine shape from DB).
    :return: Area of input mine shape in m² (float)
    """
    source_crs, target_crs = pyproj.CRS('epsg:4326'), pyproj.CRS('epsg:3857')
    proj = pyproj.Transformer.from_proj(source_crs, target_crs, always_xy=True).transform
    shapely_shape = geoalchemy2.shape.to_shape(shape.geom)  # Convert to shapely geometry
    return transform(proj, shapely_shape).area


def transform_shapelist(shapelist, target_crs, source_crs='epsg:4326'):
    """
    Transforms shapes from epsg:4326 (or another given source_crs) to another coordinate system.

    :param shapelist: list of shapely.geometries
    :param target_crs: string of target coordinate system, like 'epsg:xxxx'
    :param source_crs: string of source coordinate system, like 'epsg:xxxx'
    :return: list of shapely.geometries (like input)
    """
    project = pyproj.Transformer.from_proj(source_crs, target_crs, always_xy=True).transform
    return [shapely.ops.transform(project, shape) for shape in shapelist]


def extract_mask_v2(wsg84_shapes, tif_file_path: str):
    """
    Uses rasterio.mask.mask to extract a mask with the given shapes from the provided tif file.
    The shapes are supposed to be in WSG84 (they will be projected to the projection of the tif_file)

    :param wsg84_shapes: A list of shapely.geometry.Polygon objects
    :param tif_file_path: A string containing the path to the tif file that is used to extract the shapes
    :return: A numpy array corresponding to input tif data, with everything masked (0) except input shapes
    """
    with rasterio.open(tif_file_path, 'r') as dataset:
        coordinate_system = dataset.meta['crs']['init']
        transformed_shapes = transform_shapelist(wsg84_shapes, coordinate_system)
        out_image, out_transform = rasterio.mask.mask(dataset, transformed_shapes, nodata=0, filled=True)
    return out_image


def create_dict_with_areas_and_preds_for_gt_shapes(ground_truth_shapes, pred_folder_path: str, label: int):
    """
    Creates output array of dictionaries (one dict for each Ground Truth mine with desired label), with projected
    area (in m²) and corresponding prediction probability values.

    :param ground_truth_shapes: List of GridShapes objects (class defined in src/utils/database.py)
    :param pred_folder_path: Path to a folder that contains the prediction boxes for one or more grid_label_ids. The
                             files within the folder are expected to have the grid_label_id as name and tiff extension.
                             Ex: `/media/WD1/grid_prediction/test_set/7/model_ref_path/prediction/probability`
    :param label: Desired mine label to retrieve in {0, 1, 2} (int)
    :return: list of dictionaries, whose length is the number of input Ground Truth mine shapes.
             NB: In each dictionary, keys are 'np_masked_pred_array' and 'gt_shape_area'
    """
    mines_dicts_array = []
    for gt_shape in ground_truth_shapes:  # Loop on GT mines
        if gt_shape.label == label:
            gt_shape_area = get_projected_area_for_shape(gt_shape)  # in m²
            tif_file = os.path.join(pred_folder_path, str(gt_shape.labeled_regions_grid_id) + '.tiff')
            # Keep only the predictions values within the GT shapes from the input prediction tif file
            np_masked_pred_array = extract_mask_v2([geoalchemy2.shape.to_shape(gt_shape.geom)], tif_file)
            mines_dicts_array.append({'np_masked_pred_array': np_masked_pred_array, 'gt_shape_area': gt_shape_area})
    return mines_dicts_array


def compute_areas_biomes(biomes_df):
    """
    Computes each biome's area in given dataframe, and then computes its proportion compared to total area.

    :param biomes_df: GeoPandas dataframe which contains WWF biomes names, IDs and geometries, restricted to area of
                      interest/prediction (keys of interest are 'BIOME_NUM', 'BIOME_NAME' and 'geometry').
    :return: areas_biomes = Panda Series object containing each biome's area in m², shape = (nb_biomes, ).
    :return: areas_biomes_percentage = Panda Series object containing each biome's ratio coverage in percentage (%),
                                       shape = (nb_biomes, ).
    """
    # Compute each biome's area
    projected_biomes_df = biomes_df.to_crs(epsg=3857)  # To CRS WGS 84 / Pseudo-Mercator for area computations
    areas_biomes = projected_biomes_df.area  # Pandas series object with resulting areas in m²
    total_area = np.sum(areas_biomes)  # Compute total area covered by geometries in "biomes_df"
    logging.info(f"Total area predicted in aoi : {total_area / 1000000} km²")

    # Compute aoi coverage ratio for each biome and display results
    areas_biomes_percentage = pd.Series(index=areas_biomes.index, dtype=np.dtype('float16'))  # Shape of "areas_biomes"
    for biome_index in areas_biomes.index:
        areas_biomes_percentage[biome_index] = float('%.1f' % (100 * areas_biomes[biome_index] / total_area))  # %
        logging.info(f"Area of biome '{biomes_df.BIOME_NAME[biome_index]}' : {areas_biomes[biome_index] / 1000000} \
                     km², i.e. {areas_biomes_percentage[biome_index]}% of the total predicted area.")
    return areas_biomes, areas_biomes_percentage


def compute_recall_for_mine_size_per_threshold(mines_dicts_array, threshold: float, size_condition):
    """
    Compute recall for given mine size classification specified by mine size boundaries given in entry.

    :param mines_dicts_array: output list of dicts created by the "create_dict_with_areas_and_predictions_for_gt_shapes"
                             function.
    :param threshold: Probability threshold value in [0:1], used to convert entry proba prediction to binary (float).
    :param size_condition: Size mine category expressed as : [min_area, max_area] in m².
    :return: Recall score for this category of mines expressed in % (float with 1 decimal).
    """
    nb_mines_total = 0  # Init total nb of mines in class size
    nb_mines_found = 0  # Init nb of mines in class size that were found in prediction
    for mine_dict in mines_dicts_array:  # Loop on GT mines
        gt_shape_area = mine_dict['gt_shape_area']
        np_masked_pred_array = mine_dict['np_masked_pred_array']
        if np.min(np_masked_pred_array) != -1:  # if no data do not take it into account
            if size_condition[0] <= gt_shape_area < size_condition[1]:
                nb_mines_total += 1
                if np.max(np_masked_pred_array) > threshold:  # Check if mine was found
                    nb_mines_found += 1
    if nb_mines_total != 0:
        recall = (nb_mines_found / nb_mines_total) * 100  # in [0 : 100] (%)
    else:
        recall = 100  # In case there is no mine to detect, set recall to 100
    return float('%.1f' % recall)  # Keep only 1 decimal


# 4) Visuals creation related functions

def plot_confusion_matrix(cm, output_fig_path, group_names=None, categories='auto', count=True, percent=True, cbar=True,
                          xyticks=True, xyplotlabels=True, sum_stats=True, figsize=None, cmap='Blues', title=None):
    """
    This function will make a pretty plot of a sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Inputs:
    - cm:              Confusion matrix to be passed in, as a numpy ndarray of shape (n_classes, n_classes).
    - output_fig_path: Path to which resulting figure will be saved (expected extension is PNG or JPG). (str)
    - group_names:     List of strings that represent the labels row by row to be shown in each square.
    - categories:      List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    - count:           If True, show the raw number in the confusion matrix. Default is True.
    - normalize:       If True, show the proportions for each category. Default is True.
    - cbar:            If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
    - xyticks:         If True, show x and y ticks. Default is True.
    - xyplotlabels:    If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    - sum_stats:       If True, display summary statistics below the figure. Default is True.
    - figsize:         Tuple representing the figure size. Default will be the matplotlib rcParams value.
    - cmap:            Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html
    - title:           Title for the heatmap. Default is None.
    """

    # Code to generate text inside each square
    blanks = ['' for _ in range(cm.size)]

    if group_names and len(group_names) == cm.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])

    # Code to generate summary stats & associated text
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cm) / float(np.sum(cm))

        # if it is a binary confusion matrix, show some more stats / metrics
        if len(cm) == 2:
            precision = cm[1, 1] / sum(cm[:, 1])
            recall = cm[1, 1] / sum(cm[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # Set figure parameters according to other input arguments
    if figsize is None:  # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:  # Do not show categories if xyticks is False
        categories = False

    # Creation of Heatmap visualization + figure saving
    plt.figure(figsize=figsize)
    sn.heatmap(cm, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    if title:
        plt.title(title, pad='10.0', fontsize=15)
    plt.savefig(output_fig_path, bbox_inches=Bbox([[-1, -1], [6, 5]]))  # [[xmin, ymin], [xmax, ymax]]
