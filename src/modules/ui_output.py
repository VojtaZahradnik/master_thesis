import os
import pandas as pd
from src.modules import evl, conf
import folium
import gpxpy
from datetime import datetime

def create_html() -> str:
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MT report</title>
        <h1 class="centered"><b>{activity_name}</b></h1>
    <style>
        .container {{
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }}
        img {{
            max-width: 80%;
            height: auto;
            border: 2px solid black; 
            border-radius: 20px;
        }}
        h2 {{
            display: block;
        }}
        table {{
            border-collapse: collapse;
            width: 80%;
            margin: 20px 0;
            margin-left: auto;
            margin-right: auto;
        }}
        th, td {{
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        h1.centered {{
            text-align: center;
        }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Cadence of the activity</h1>
            <img src="data:image/png;base64, {cad_plot}" alt="Cadence Plot Image">
        </div>
    
        <div class="container">
            <h1>Heart rate of the activity</h1>
            <img src="data:image/png;base64, {hr_plot}" alt="Heart Rate Plot Image">
        </div>
    
        <div class="container">
            <h1>Enhanced speed of the activity</h1>
            <img src="data:image/png;base64, {speed_plot}" alt="Speed Plot Image">
        </div>
        
         <h1 class="centered">{final_time}</h1>
         
         <hr>
    
    <table>
        <tr>
            <th><b>Average cadence</b></th>
            <td>{avg_speed} km/h</td>
        </tr>
        <tr>
            <th><b>Average heart rate</b></th>
            <td>{avg_hr} bpm</td>
        </tr>
        
        <tr>
            <th><b>Average speed</b></th>
            <td>{avg_cadence} rpm</td>
        </tr>
        
        <tr>
            <th><b>Average stride length</b></th>
            <td>{avg_sl} cm</td>
        </tr>
        
        <tr>
            <th><b>Average temperature</b></th>
            <td>{avg_tmp} C</td>
        </tr>
        
        <tr>
            <th><b>Total Ascent </b></th>
            <td>{ascent} m</td>
        </tr>
        
        <tr>
            <th><b>Total Descent </b></th>
            <td>{descent} m</td>
        </tr>
        
    </table>
    
    <hr>
    
    <h1 class="centered">Compare</h1>
    
     <div class="container">
            <img src="data:image/png;base64, {radar_plot}" alt="Radar Plot Image">
        </div>
        
         <div class="container">
            <h1>Bechovice Praha</h1>
            <img src="data:image/png;base64, {bechovice_plot}" alt="Bechovice Praha Plot Image">
        </div>
        
         <div class="container">
            <h1>Halfmarathon</h1>
            <img src="data:image/png;base64, {hradec_plot}" alt="Hradec Kralove Plot Image">
        </div>
        
         <div class="container">
            <h1>Boston Marathon</h1>
            <img src="data:image/png;base64, {boston_plot}" alt="Boston Marathon Plot Image">
        </div>
    
    
        <hr>
         <h1 class="centered">Map of the Race</h1>
        <div class="container">
            <iframe src="../maps/map_{track_name}.html" width="80%" height="600" frameborder="0" scrolling="no"></iframe>
        </div>

    </body>
    </html>
    """

    return html


def save_report(athlete_name: str,
                activity_type: str,
                activity_name: str,
                cad_plot: str,
                hr_plot: str,
                speed_plot: str,
                final_time: str,
                df,
                radar_plot,
                track_name: str):
    with open(os.path.join("reports", f"report_{athlete_name}_{activity_name.lower()}_{datetime.now().strftime('%Y%m%d%M')}.html"), "w") as html_file:
        html_file.write(create_html()
                        .format(activity_name=activity_name,
                                cad_plot=cad_plot,
                                hr_plot=hr_plot,
                                speed_plot=speed_plot,
                                final_time = final_time,
                                avg_speed=round(df["enhanced_speed"].mean()),
                                avg_hr=round(df["heart_rate"].mean()),
                                avg_cadence=round(df["cadence"].mean()),
                                avg_tmp=round(df["temp"].mean(),1),
                                avg_sl=round(calc_stride_length(max(df.distance),df.cadence.mean()),1),
                                ascent=round(sum(df["slope_ascent"])),
                                descent=round(sum(df["slope_descent"])),
                                radar_plot = radar_plot,
                                bechovice_plot = ref_track(track_name="bechovice"),
                                hradec_plot = ref_track(track_name="hradec"),
                                boston_plot = ref_track(track_name="boston"),
                                track_name=track_name))

    print("LOG: HTML page with the plots next to each other has been created.")

def ref_track(track_name: str, ref_athlete_name = "zimola"):
    pred_1 = pd.read_csv(f"src/{conf['Paths']['output']}/{conf['Athlete']['name']}_{track_name}.csv")
    pred_2 = pd.read_csv(f"src/{conf['Paths']['output']}/{ref_athlete_name}_{track_name}.csv")

    plot = evl.plot_compare(df=pred_1, pred1=pred_1["enhanced_speed"], pred2=pred_2["enhanced_speed"])

    return plot

def calc_stride_length(distance: pd.Series, cadence: pd.Series):
    return distance/cadence

def gen_map(track_name: str):
    # Read the GPX file and parse coordinates
    gpx_file = f"tracks/{track_name}.gpx"
    gpx_data = open(gpx_file, "r").read()
    gpx_parser = gpxpy.parse(gpx_data)

    # Extract coordinates from GPX data
    coordinates = []
    for track in gpx_parser.tracks:
        for segment in track.segments:
            for point in segment.points:
                coordinates.append([point.latitude, point.longitude])

    # Create a folium map centered around the first coordinate
    map_center = [coordinates[0][0], coordinates[0][1]]
    mymap = folium.Map(location=map_center, zoom_start=14)

    # Add polyline to the map
    folium.PolyLine(locations=coordinates, color="blue").add_to(mymap)

    # Save the map as an HTML file
    mymap.save(f"maps/map_{track_name}.html")