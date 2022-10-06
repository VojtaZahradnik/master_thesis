import os
from pathlib import Path

from _Config import _Config
from _Logger import _Logger

# Some tricky stuff for path handling, opt. in the future
# For everyone is the same path something/CallOutBot/
abspath = os.path.abspath(__file__)
os.chdir(os.path.dirname(abspath))
p = Path(os.getcwd())
os.chdir(p.parent.parent)

log = _Logger(project_name="master_thesis")
log.create_log(name="master_thesis.log", dir=os.getcwd(), dir_name="logs")
log = log.get_logger()

conf = _Config(config_name="config.yaml").get_config_file()
