import os
from pathlib import Path

# Some tricky stuff for path handling, opt. in the future
# For everyone is the same path something/master_thesis/
import sys
sys.path.insert(0,'/home/zahradnik/PycharmProjects/master_thesis')

abspath = os.path.abspath(__file__)
print(abspath)
os.chdir(os.path.dirname(abspath))
p = Path(os.getcwd())
os.chdir(p.parent.parent)
print(os.getcwd())

from src._Config import _Config
from src._Logger import _Logger

log = _Logger(project_name="master_thesis")
log.create_log(name="master_thesis.log", dir=os.getcwd(), dir_name="logs")
log = log.get_logger()

conf = _Config(config_name="config.yaml").get_config_file()


df_columns = ['timestamp','heart_rate','distance','enhanced_altitude','cadence','enhanced_speed']
compare_features = ["enhanced_altitude","temp","rain","heart_rate","cadence","slope_ascent","slope_descent"]