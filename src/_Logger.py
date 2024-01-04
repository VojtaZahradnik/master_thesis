import logging
import os


class _Logger:
    """
    Class for implementation of logging class
    """

    def __init__(self, project_name: str):
        """
        Loading config variables
        """
        self.project_name = project_name

    def create_log(self, name: str, dir: str, dir_name: str):
        """
        Basic setup of logger
        :param name: Name of the logger
        :param dir: Path to directory, where we have logs
        :param dir_name: Name of directory, where we have logs
        """
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
        """
        Returning logger object
        :return: Object of logging library
        """
        return logging.getLogger(self.project_name)
