# Standard imports
import datetime
import logging
import os


def setup_logging_for_evaluation(eval_folder_path: str, name_cmd: str):
    """
    Setup the logging configuration according to the evaluate command called.

    :param str eval_folder_path: Path to the evaluation folder that stores outputs (figs, tables).
    :param str name_cmd: Name of the evaluate command used to compute metrics (Ex: 'all-metrics').
    """
    log_file = os.path.join(eval_folder_path, 'log_eval_{}_{}.log'.format(name_cmd,
                                                                          datetime.datetime.now().strftime('%d-%m-%Y')))
    if os.path.exists(log_file):
        print('Warning: log file already exists at {}\nNew logs will be appended to its end.'.format(log_file))
    else:
        print('Logs will be saved to {}'.format(log_file))
    logging.basicConfig(filename=log_file, filemode='a', format='%(levelname)s - %(asctime)s | %(message)s',
                        datefmt='%H:%M:%S', level=logging.INFO, force=True)
    logging.info('Start of the evaluation.')


def setup_logging_for_prediction(output_folder_path: str, name_cmd: str):
    """
    Setup the logging configuration according to the evaluate command called.

    :param str output_folder_path: Path to the folder that stores prediction's outputs.
    :param str name_cmd: Name of the evaluate command used to compute metrics (Ex: 'all-metrics').
    """
    log_file = os.path.join(output_folder_path, 'log_{}_{}.log'.format(name_cmd,
                                                                       datetime.datetime.now().strftime('%d-%m-%Y')))
    if os.path.exists(log_file):
        print('Warning: log file already exists at {}\nNew logs will be appended to its end.'.format(log_file))
    else:
        print('Logs will be saved to {}'.format(log_file))
    logging.basicConfig(filename=log_file, filemode='a', format='%(levelname)s - %(asctime)s | %(message)s',
                        datefmt='%H:%M:%S', level=logging.INFO, force=True)
    logging.info('Start of the evaluation.')