import glob
import os
import shutil
import threading
import time

import fitparse
import pyunpack
from src.modules import conf, log
from tqdm import tqdm

"""
Basic extract and sort of fit files.
"""

def unpack(path_to_load: str, path_to_save: str, athlete_name: str):
    """
    Unpack fit files from compressed format.
    :param path_to_load: Pickle file path.
    :param path_to_save: Save path of sorted zahradnik.
    :param athlete_name: Name of the athlete.
    """
    start = time.monotonic()
    [os.remove(file) for file in glob.glob(os.path.join(conf['Paths']['fit'], conf['Athlete']['name'], "*.fit"))]

    files = glob.glob(os.path.join(path_to_load, athlete_name, "*.zip"))
    if(files == []):
        files = glob.glob(os.path.join(path_to_load, athlete_name, "*.gz"))
    if path_to_save not in os.listdir():
        os.mkdir(path_to_save)
    if athlete_name not in os.listdir(path_to_save):
        os.mkdir(os.path.join(path_to_save, athlete_name))
    if len(files) != 0:
        for x in tqdm(range(len(files))):
            pyunpack.Archive(files[x]).extractall(
                os.path.join(path_to_save, athlete_name)
            )

        log.info(
            f"{len(files)} files unpacked after {round(time.monotonic() - start, 2)} seconds"
        )

    else:
        log.warning(f"Raw data {path_to_load} folder for unpack is empty")

def sort_activities(athlete_name: str, path_to_save: str):
    """
    Sort zahradnik by type and will choose zahradnik with needed variables
    :param athlete_name: Name of the athlete
    :param path_to_save: save path of sorted zahradnik
    """
    files = glob.glob(os.path.join(conf['Paths']['fit'], athlete_name, "*.fit"))
    [shutil.rmtree(file) for file in glob.glob(os.path.join(conf['Paths']['fit'], athlete_name, "*")) if os.path.isdir(file)]

    if(len(files)!=0):
        start = time.monotonic()
        for x in tqdm(range(len(files))):
            fitfile = fitparse.FitFile(files[x])
            for record in fitfile.get_messages("session"):
                if (
                    record.get_value("sub_sport") != "indoor_cycling"
                    and record.get_value("sub_sport") != "treadmill"
                ):
                    activity_type = record.get_value("sport")
                    if activity_type not in os.listdir(
                        os.path.join(path_to_save, athlete_name)
                    ):
                        os.mkdir(
                            os.path.join(
                                path_to_save,
                                athlete_name,
                                activity_type,
                            )
                        )
                    shutil.copyfile(
                        files[x],
                        os.path.join(
                            path_to_save,
                            athlete_name,
                            activity_type,
                            os.path.split(files[x])[-1],
                        ),
                    )
            os.remove(files[x])
        log.info(
            f"{len(files)} files sorted after {round(time.monotonic() - start, 2)} seconds"
        )
    else:
        log.warning(f"{athlete_name} folder for fits is empty.")