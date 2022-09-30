# ETL
<p align="center">
  <img src="../../research_task/res/etl.png" />
</p>

Extract, transform, load process is implemented mainly in [raw_data.py](../../research_task/src/modules/raw_data.py) and [preprocess.py](../../research_task/src/modules/preprocess.py). 
These two modules provide many unnecessary functions:

## Unpacking

Unpacking is implemented in [raw_data.py](../../research_task/src/modules/raw_data.py). From [Strava](https://www.strava.com) we get FIT and GPX files in zip file format. We need to open it and unpack. 
For this purpose I used Python library [pyunpack](https://pypi.org/project/pyunpack/).

## Sorting activities

After unpacking, we have many FIT files in one group. We need to sort them into smaller groups based on activity type. 
For opening FIT files I used library called [fitparse](https://pypi.org/project/fitparse/). I am sorting activity based on activity type and dropping the indoor cycling and treadmill runs. Reading FIT files 
is really time-consuming and the whole process has big time complexity.

## Outlier detection

Some records are outliers, what is observation that lies an abnormal distance from other values. I am detecting them with thresholds for variables. 
For example every observation of heart rate, that is above threshold of 210 bpm, is dropped.

## Enrichment

From FIT files we have few exogenous variables, but we need more. For this reason we need to enrich the dataset. 
We enchrich data in many category:

- **Meteo**: We add weather condition data into dataset. For example rain, wind direct, wind speed etc.
- **Slope**: We calculate slope ascent, descent and percent of slope.
- **Delayed variables**: We calculate also delayed variable, mainly cadence.

## Saving into Pickles

We are saving enriched data into Pickles. Pickles was chosen, because from elementary file formats was best suited for this problem. Database would be better for sure, but 
for now is just over kill and time waste. Loading from FIT files was very time complex, so we needed change the format.

