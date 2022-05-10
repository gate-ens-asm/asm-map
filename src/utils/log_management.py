# Standard imports
import logging, os, re, datetime


def setup_logging_for_evaluation(eval_folder_path: str, s2mines_cmd: str):
    """
    Setup the configuration for logging for the s2mines evaluate command called.

    :param eval_folder_path: Path to the evaluation folder that stores outputs (figs, tables).
                             NB: /!\ The folder path is supposed to contain the dataset_type and dataset_id infos, as:
                             `/media/MONOD3/grid_prediction/ds_type/ds_id/mlflow/mlfflow_id/model_id/prediction/probability`
    :param s2mines_cmd: Name of the s2mines evaluate command used to compute metrics (Ex: 'all-metrics').
    """
    # Read evaluated dataset's type / id
    try:
        eval_path_match = re.search('(training|validation|test)/([0-9][0-9]?)/', eval_folder_path)
        ds_type = eval_path_match.group(1)
        ds_id = int(eval_path_match.group(2))
    except Exception as e:
        print('ERROR: Prediction folder path does not respect the naming conventions, dataset type/ID not found :\n', e)
        return -1

    # Configure logging and initiate logfile
    log_file = os.path.join(eval_folder_path, 'log_eval_{}_{}.log'.format(s2mines_cmd,
                                                                          datetime.datetime.now().strftime('%d-%m-%Y')))
    if os.path.exists(log_file):
        print('Warning: log file already exists at {}\nNew logs will be append to its end.'.format(log_file))
    else:
        print('Logs will be saved to {}'.format(log_file))
    logging.basicConfig(filename=log_file, filemode='a', format='%(levelname)s - %(asctime)s | %(message)s',
                        datefmt='%H:%M:%S', level=logging.INFO, force=True)
    logging.info('Start of the evaluation.')
    logging.info('Dataset evaluated : {} set {}\n'.format(ds_type, ds_id))
