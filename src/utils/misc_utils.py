"""
Miscellaneous utils functions.
"""

# Standard imports
import logging

# Third-party imports
import yaml


def load_config(config_file: str):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error('Could not read yaml configuration file: ', exc)
            return -1
    return config
