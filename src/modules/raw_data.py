import glob
import logging
import os
import shutil
import time

import fitparse
import pyunpack
from tqdm import tqdm

"""
Basic extract and sort of Fit files
"""


def input_cli(conf: dict, log: logging.Logger):
    """
    Solution for CLI input without __init__
    :param conf: Configuration dictionary
    """
    unpack(path_to_load=conf["Paths"]["raw"], path_to_save=conf["Paths"]["fit"])
    log.info("Unpacked")
    sort_activities(athlete=conf["Athlete"]["name"], path_to_save=conf["Paths"]["fit"])
    log.info("Sorted")


def unpack(path_to_load: str, path_to_save: str):
    """
    Unpack fit files from compressed format
    :param path_to_load: pickle file path
    :param path_to_save: save path of sorted activities
    """
    start = time.monotonic()
    files = glob.glob(f"{path_to_load}/*.gz")
    if len(files) != 0:
        for x in tqdm(range(len(files))):
            _, ext = os.path.splitext(files[x])
            pyunpack.Archive(os.path.join(path_to_load, files[x])).extractall(
                path_to_save
            )
            os.remove(os.path.join(path_to_load, files[x]))
        logging.getLogger("project").info(
            f"{len(files)} files unpacked after {round(time.monotonic() - start, 2)}"
        )

    else:
        logging.getLogger("project").warning("Raw data folder for unpack is empty")


def sort_activities(athlete: str, path_to_save: str):
    """
    Sort activities by type and will choose activities with needed variables
    :param athlete: name of the athlete
    :param path_to_save: save path of sorted activities
    """
    start = time.monotonic()
    if athlete not in os.listdir(path_to_save):
        os.mkdir(os.path.join(path_to_save, athlete))
    files = [x for x in os.listdir(path_to_save) if ".fit" in x]
    if len(files) != 0:
        for x in tqdm(range(len(files))):
            if not (files[x].isnumeric()):
                continue
            name, ext = os.path.splitext(files[x])
            fitfile = fitparse.FitFile(os.path.join(path_to_save, files[x]))

            for record in fitfile.get_messages("session"):
                if (
                    record.get_value("sub_sport") != "indoor_cycling"
                    and record.get_value("sub_sport") != "treadmill"
                ):
                    if record.get_value("sport") not in os.listdir(
                        path_to_save + athlete
                    ):
                        os.mkdir(
                            os.path.join(
                                path_to_save,
                                athlete,
                                record.get_value("sport"),
                            )
                        )
                    shutil.copyfile(
                        os.path.join(path_to_save, files[x]),
                        os.path.join(
                            path_to_save,
                            athlete,
                            record.get_value("sport"),
                            files[x],
                        ),
                    )
            os.remove(os.path.join(path_to_save, name + ext))
        logging.getLogger("project").info(
            f"{len(files)} files sorted after {round(time.monotonic() - start, 2)}"
        )
    else:
        logging.getLogger("project").warning("Datasets folder is empty.")
