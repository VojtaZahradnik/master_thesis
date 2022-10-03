import argparse
import os
from pathlib import Path

import modules
from modules import log,conf

def main():
    """
    Implementation of argparse library for user-friendly CLI
    """
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument(
        "--module",
        dest="module",
        type=str,
        help="Module name you want to call",
        default=None,
    )
    args = my_parser.parse_known_args()

    eval(vars(args[0])["module"]).input_cli(conf=conf, log=log)

    my_parser.set_defaults()


if __name__ == "__main__":
    log.info('dasdas')
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     log.error("Keyboard Interrupt")