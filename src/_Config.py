import sys

import yaml


class _Config:
    def __init__(self, config_name: str, log_name: str):
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
        if property_name not in self.conf.keys():
            return None
        return self.conf[property_name]

    def get_config_file(self) -> dict:
        return self.conf

    def append_row(self, key: str, value: int):
        self.conf[key] = value

    def save_conf_file(self):
        with open(self.config_name, "w") as outfile:
            yaml.dump(self.conf, outfile, default_flow_style=False)
