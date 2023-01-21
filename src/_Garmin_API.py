import os
import datetime
from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
    GarminConnectAuthenticationError,
)
import glob
from src.modules import conf, fit, raw_data, preprocess, df_columns
from tqdm import tqdm

class _GarminAPI:
    """
    Class for download activities files from Garmin through GarminAPI
    """

    def __init__(self, athlete_name: str):
        """
        Initial function to determine basic variables
        """
        self.athlete_name = athlete_name

        if(conf['Paths']['raw'] not in os.listdir()):
            os.mkdir(conf['Paths']['raw'])

        if(self.athlete_name not in os.listdir(conf['Paths']['raw'])):
            os.mkdir(os.path.join(conf['Paths']['raw'],self.athlete_name))
        else:
            [os.remove(file) for file in glob.glob(os.path.join(conf['Paths']['raw'], self.athlete_name,"*"))]

        self.end_date = None
        self.start_date = None
        self.api = Garmin(conf['Garmin']['email'], conf['Garmin']['pass'])
        self.api.login()

    def set_range(self, days_to_past: int, end_date=datetime.date.today()):
        """
        Function to determine date range for activities
        :param days_to_past: How many days we want to be in the past
        :param end_date: End date of activities
        """
        self.start_date = datetime.date.today() - datetime.timedelta(days=days_to_past)
        self.end_date = end_date

    def download_data(self):
        """
        Function to download fit files from Garmin, going through all activities in date range
        """
        activities = self.api.get_activities_by_date(self.start_date, self.end_date)
        for activity in tqdm(activities):
            activity_id = activity["activityId"]
            zip_data = self.api.download_activity(activity_id,
                                             dl_fmt=self.api.ActivityDownloadFormat.ORIGINAL)
            output_file = os.path.join(conf['Paths']['raw'], self.athlete_name, f'{str(activity_id)}.zip')
            with open(output_file, "wb") as fb:
                fb.write(zip_data)

