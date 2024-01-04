import argparse

from modules import conf, fit, log, preprocess, ui_output, evl
from modules.compare import Compare
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from feature_engine.creation import MathFeatures
import time
import os
from datetime import datetime, timedelta

def main():
    """
    Implementation of argparse library for user-friendly CLI
    """
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument(
        "--gpx_file",
        dest="gpx_file",
        type=str,
        help="Day of the competition",
        default=None,
    )
    my_parser.add_argument(
        "--race_day",
        dest="race_day",
        type=str,
        help="Name of the GPX file in tracks folder in format YYYY-mm-dd",
        default=None,
    )
    args = my_parser.parse_args()
    dict_param = vars(args)
    print(f"LOG: Params: {dict_param}")
    my_parser.set_defaults()

    start = time.monotonic()
    print("LOG: Starting application")
    data = fit.load_pcls(
        'zahradnik',
        'running',
        conf["Paths"]["pcl"],
        race_day=  datetime.strptime(dict_param["race_day"], "%Y-%m-%d")
    )
    print(f"LOG: {len(data)} activities loaded")

    test_df = preprocess.load_test_activity(path=os.path.join(conf["Paths"]["test_activities"],dict_param["gpx_file"]),
                                            race_day=f"{dict_param['race_day']}-11-30")

    print(f"LOG: Activity {dict_param['gpx_file']} successfully loaded with length of {len(test_df)} rows")

    low_dist, high_dist = preprocess.segment_data(data)

    if max(test_df.distance) > 10000:
        train_df = fit.clean_data(pd.concat(high_dist))
    else:
        train_df = fit.clean_data(pd.concat(low_dist))

    clf = XGBRegressor()
    print("LOG: Starting training of models")

    # CADENCE
    clf.fit(train_df[test_df.columns], train_df.cadence)
    test_df['cadence'] = clf.predict(test_df)
    print("LOG: Cadence model trained successfully")
    print(f"LOG: Mean of cadence: {test_df['cadence'].mean()}")

    test_df = preprocess.calc_windows(df=test_df,
                                      lagged=15,
                                      cols=["cadence"])
    test_df = preprocess.calc_moving(df=test_df, max_range=110, col="cadence")

    # HEART RATE
    clf.fit(train_df[test_df.columns], train_df.heart_rate)
    test_df["heart_rate"] = clf.predict(test_df)
    print("LOG: Heart rate model trained successfully")
    print(f"LOG: Mean of heart rate: {test_df['heart_rate'].mean()}")

    for fce in ["sum", "mean", "min", "max"]:
        test_df = (MathFeatures(variables=["heart_rate", "cadence"], func=fce)
                   .fit(test_df).transform(test_df))

    # ENHANCED_SPEED
    test_df= preprocess.calc_windows(df=test_df,
              lagged=12,
              cols=["heart_rate"])
    test_df = preprocess.calc_moving(df=test_df, max_range=110, col="heart_rate")
    clf.fit(train_df[test_df.columns], train_df.enhanced_speed)
    test_df["enhanced_speed"] = clf.predict(test_df)
    print("LOG: Speed model trained successfully")
    print(f"LOG: Mean of speed: {test_df['enhanced_speed'].mean()}")

    test_df['enhanced_speed'] = [x if x > 15 else np.mean(test_df['enhanced_speed']) or x if x < 30 else np.mean(test_df['enhanced_speed']) for x in
                    test_df['enhanced_speed']]
    final_time = (fit
                  .calc_final_time(distance=test_df["distance"],
                           speed = test_df['enhanced_speed']))
    print(f"LOG: {final_time}")

    percent_delay = int(len(test_df) * 0.02)  ## 2% delay

    cad_plot = evl.plot(df=test_df[percent_delay:], pred=test_df['cadence'][percent_delay:],
                        ylabel="Cadence", color="green")
    hr_plot = evl.plot(df=test_df[percent_delay:], pred=test_df['heart_rate'][percent_delay:],
                       ylabel="Heart rate", color="red")
    speed_plot = evl.plot(df=test_df[percent_delay:], pred=test_df["enhanced_speed"][percent_delay:],
                          ylabel="Speed", color="blue")

    next_day = f"{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}-11-30"
    print(f"LOG: Loading reference tracks for {next_day}")

    bechovice = preprocess.load_test_activity(path="tracks/bechovice.gpx",
                                               race_day=next_day)

    fit.get_final_df(train_df=train_df,
                     test_df=bechovice,
                     model=XGBRegressor(),
                     race_name="bechovice",
                     athlete_name=conf["Athlete"]["name"])
    print("LOG: Model based on Bechovice track successfully trained")

    hradec = preprocess.load_test_activity(path="tracks/hradec_half.gpx",
                                           race_day=next_day)
    fit.get_final_df(train_df=train_df,
                     test_df=hradec,
                     model=XGBRegressor(),
                     race_name="hradec",
                     athlete_name=conf["Athlete"]["name"])
    print("LOG: Model based on Hradec Kralove halfmarathon track successfully trained")


    boston = preprocess.load_test_activity(path="tracks/boston.gpx",
                                           race_day=next_day)
    fit.get_final_df(train_df=train_df,
                     test_df=boston,
                     model=XGBRegressor(),
                     race_name="boston",
                     athlete_name=conf["Athlete"]["name"])
    print("LOG: Model based on Boston marathon track successfully trained")

    compare = Compare(ref_athlete_name="zimola")
    compare.calc_importances(cols=test_df.columns, clf=clf)
    radar = compare.plot_radar()
    print("LOG: Model comparison and radar plot generated")

    track_name = dict_param["gpx_file"].replace(".gpx","")

    ui_output.gen_map(track_name=track_name)
    print(f"LOG: Map generated for track: {track_name}")
    ui_output.save_report(athlete_name=conf["Athlete"]["name"],
                          activity_type=conf["Athlete"]["activity_type"],
                          activity_name="bechovice.gpx"
                          .replace(".gpx", "")
                          .capitalize(),
                          cad_plot=cad_plot,
                          hr_plot=hr_plot,
                          speed_plot=speed_plot,
                          final_time=final_time,
                          df=test_df,
                          radar_plot=radar,
                          track_name=track_name
                          )
    print("LOG: Report saved")

    print(f"LOG: Final time of run: {time.monotonic() - start} second")

if __name__ == "__main__":
    try:
        print("LOG: Starting application")
        main()
        print("LOG: Ending application")
    except KeyboardInterrupt:
        log.error("Keyboard Interrupt")
