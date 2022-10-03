import logging
import os


class _Logger:
    def __init__(self, project_name: str):
        self.project_name = project_name

    def create_log(self, name: str, dir: str, dir_name: str):

        if not (dir_name in os.listdir(dir)):
            os.mkdir(os.path.join(dir, dir_name))

        logger = logging.getLogger(self.project_name)

        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(dir, dir_name, name), mode="a")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(self.project_name)
