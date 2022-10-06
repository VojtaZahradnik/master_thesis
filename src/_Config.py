import sys

import yaml


class _Config:
    """
    Configuration class, that handle config file in format yaml
    """
    def __init__(self, config_name: str):
        """
        Initialize function to load config file
        :param config_name: Name of the config file
        """
        self.config_name = config_name
        from modules import log

        try:
            with open(self.config_name) as f:
                self.conf = yaml.load(f, Loader=yaml.FullLoader)
            log.info(f"Config file {config_name} successfully loaded")

        except FileNotFoundError:
            log.info("Config file not found")
            sys.exit()

    def get_property(self, property_name):
        """
        Function that will find value from property name in config file
        :param property_name: Name of the key
        :return: Value based on key
        """
        if property_name not in self.conf.keys():
            return None
        return self.conf[property_name]

    def get_config_file(self) -> dict:
        """
        Transfer config file to dictionary
        :return: Dictionary of config file
        """
        return self.conf

    def append_row(self, key: str, value: int):
        """
        Append row to config file
        :param key: Key in the dict.
        :param value: Value in the dict.
        """
        self.conf[key] = value

    def save_conf_file(self):
        """
        Method for save new config file
        """
        with open(self.config_name, 'w') as outfile:
            yaml.dump(self.conf, outfile, default_flow_style=False)