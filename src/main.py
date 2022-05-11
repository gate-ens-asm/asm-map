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


# MAIN
if __name__ == '__main__':
    # Read configuration file
    asm_map_folder = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(asm_map_folder, 'config', 'config.yaml')
    example_config_path = os.path.join(asm_map_folder, 'config', 'example_config.yaml')
    config = load_config(config_path)

    # Launch corresponding command
    print(f'{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- ASM MAP SCRIPT LAUNCHED --\n')
    if str(config['command']).lower() == 'predict':
        print(f'{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- PREDICTION LAUNCHED --\n')
        # TODO
    elif str(config['command']).lower() == 'evaluate':
        print(f'{datetime.datetime.now().strftime("%d-%m-%Y %H:%M")} -- EVALUATION LAUNCHED --\n')
        # TODO
    else:
        print(f'Command arg in the configuration file not recognized: {config["command"]}\n')
        print('Please specify command as either "predict" or "evaluate", then launch this script again.\n')
        print(f'For your information, you can find the configuration file at : {config_path}')
        print(f'For your information, an example of configuration file is available at : {example_config_path}')
