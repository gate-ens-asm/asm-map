"""
Miscellaneous utils functions.
"""

# Standard imports
import os

# Third-party imports
import yaml


def load_config(config_file: str):
    """
    Read the input configuration file's content (expected file format is "xxx.yaml").

    :param str config_file: Path to the configuration file.
    :return: dict with configuration parameters.
    """
    if not os.path.isfile(config_file):
        raise FileNotFoundError('Configuration file does not exist at this location : {}\n'.format(config_file))
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise Exception('Could not read YAML configuration file: {}\n'.format(exc))
    return config
