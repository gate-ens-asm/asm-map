"""
Main script, to launch the asm-map applications (prediction and evaluation).
"""

# Standard imports
import sys
import os
import datetime

# Third-party imports

# Local imports
sys.path.append('.')
from src import evaluate as evl
from src import predict as pred
from src.utils.misc_utils import load_config


# Main functions definition
def launch_prediction(params_dict: dict):
    """
    Launches the right prediction script from src/predict.py according to command_name, if all params pass controls.

    :param dict params_dict: Dictionary containing the needed parameters for prediction.
    """
    # Read configuration file
    try:
        cmd_name = 'predict_{}'.format(params_dict['subcommand'])
        if cmd_name not in dir(pred):
            raise ValueError(f'Sub-command name not recognized. {cmd_name} does not exist in "src/predict.py" module.')
        model_path = str(params_dict['model_path'])
        images_folder_path = str(params_dict['tifs_folder_path'])
        output_folder_path = str(params_dict['output_folder_path'])
        binary_threshold = float(params_dict['binary_threshold'])
    except Exception as e:
        print('ERROR: Input parameters do not fit expectations :\n', e)
        print('Try to modify the configuration file accordingly and launch again. Aborting.')

    # Launch the right prediction command
    if cmd_name == 'predict_dataset':
        out = pred.predict_dataset(model_path, images_folder_path, output_folder_path, binary_threshold)

    # Manage ending
    if out is not None and out == -1:
        print(f'\n{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- Prediction was aborted. Ending.')
    else:
        print(f'\n{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- Prediction was successfully performed.')
        print('Results are available at : ', output_folder_path)


def launch_evaluation(params_dict: dict):
    """
    Launches the right evaluation script from src/evaluate.py according to command_name, if all params pass controls.

    :param dict params_dict: Dictionary containing the needed parameters for evaluation.
    """
    # Read configuration file
    try:
        cmd_name = 'evaluate_{}_metric'.format(params_dict['subcommand'])
        if cmd_name not in dir(evl):
            raise ValueError(f'Sub-command name not recognized. {cmd_name} does not exist in "src/evaluate.py" module.')

        gt_folder_path = str(params_dict['gt_folder_path'])
        pred_folder_path = str(params_dict['pred_folder_path'])
        eval_folder_path = str(params_dict['eval_folder_path'])
        gt_asm_shapefile = str(params_dict['gt_asm_shapefile'])
        if params_dict['aoi_shapefile'] is not None:
            aoi_shapefile = str(params_dict['aoi_shapefile'])
        else:
            aoi_shapefile = ''
        if params_dict['no_mcc'] is not None:
            no_mcc_bool = bool(params_dict['no_mcc'])
        if params_dict['no_fscore'] is not None:
            no_fscore_bool = bool(params_dict['no_fscore'])
        if params_dict['output_filename'] is not None:
            fig1_filename = str(params_dict['output_filename'])
    except Exception as e:
        print('ERROR: Input parameters do not fit expectations : ', e)
        print('Try to modify the configuration file and launch again. Aborting.')

    # Launch the right evaluation command
    if cmd_name == 'evaluate_all_metric':
        out = evl.evaluate_all_metric(pred_folder_path, gt_folder_path, eval_folder_path, gt_asm_shapefile,
                                      aoi_shapefile)
    elif cmd_name == 'evaluate_threshold_metric':
        out = evl.evaluate_threshold_metric(pred_folder_path, gt_folder_path, eval_folder_path, no_mcc_bool,
                                            no_fscore_bool, fig1_filename)
    elif cmd_name == 'evaluate_size_metric':
        out = evl.evaluate_size_metric(pred_folder_path, gt_asm_shapefile, eval_folder_path)
    elif cmd_name == 'evaluate_biomes_metric':
        out = evl.evaluate_biomes_metric(pred_folder_path, gt_folder_path, eval_folder_path, aoi_shapefile)
    elif cmd_name == 'evaluate_roc_metric':
        out = evl.evaluate_roc_metric(pred_folder_path, gt_folder_path, eval_folder_path)
    elif cmd_name == 'evaluate_matrix_metric':
        out = evl.evaluate_matrix_metric(pred_folder_path, gt_folder_path, eval_folder_path)

    # Manage ending
    if out is not None and out == -1:
        print(f'\n{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- Evaluation was aborted. Ending.')
    else:
        print(f'\n{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- Evaluation was successfully performed.')
        print('Metrics are available at : ', eval_folder_path)


# MAIN
if __name__ == '__main__':
    # Read configuration file
    asm_map_folder = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(asm_map_folder, 'config', 'config.yaml')
    example_config_path = os.path.join(asm_map_folder, 'config', 'example_config.yaml')
    config = load_config(config_path)

    # Launch corresponding command
    print(f'{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- ASM MAP SCRIPT LAUNCHED\n')
    if str(config['command']).lower() == 'predict':
        print(f'{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- PREDICTION LAUNCHED\n')
        launch_prediction(config['prediction'])

    elif str(config['command']).lower() == 'evaluate':
        print(f'{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- EVALUATION LAUNCHED\n')
        launch_evaluation(config['evaluation'])

    else:
        print(f'Command arg in the configuration file not recognized: {config["command"]}\n')
        print('Please specify command as either "predict" or "evaluate", then launch this script again.\n')
        print(f'For your information, you can find the configuration file at : {config_path}')
        print(f'For your information, an example of configuration file is available at : {example_config_path}')
