from flask import Flask,render_template, request, url_for
from os import getcwd
import pandas as pd

app = Flask(__name__, static_folder='static')
data_path = 'data/'
webcam_url_base = 'https://video.autostrade.it/video-mp4_hq'
webcam_url_base = 'static/'
df_cameras = pd.read_csv(data_path + 'cameras.csv')
extension = 'mp4'

@app.route('/')
def home():
    points = getMapPoints()
    return render_template('index.html', points = points)

@app.route('/dashboard')
def specific():
    cam_code = request.args.get('cam_code')
    data, loc_name, direction, lat, long = getStatistics(cam_code)
    road = df_cameras[df_cameras['cam_code']==cam_code]['road'].iloc[0]
    return render_template('camera_feed_dashboard.html', data = data, loc_name = loc_name, direction = direction, lat = lat, long = long, camURL = f'{webcam_url_base}c{cam_code[5:]}.{extension}', road = road)

def getMapPoints():
    df_active_cameras = df_cameras[df_cameras['active']==True]
    points = [
    {"coords": [row["lat"], row["long"]], "url": request.url + 'dashboard?cam_code=' + row["cam_code"]}
    for _, row in df_active_cameras.iterrows()]
    return points

def getStatistics(cam_code):
    camera = df_cameras[df_cameras['cam_code']==cam_code]
    df_data_cam = pd.read_csv(data_path + 'datacollection.csv')
    df_data_cam = df_data_cam[df_data_cam['cam_code']==cam_code]
    df_data = df_data_cam
    df_data['date'] = pd.to_datetime(df_data['date'],format='%Y-%m-%d_%H-%M-%S')
    df_data = df_data.set_index('date')
    df_data = df_data.drop('cam_code', axis=1)
    out = df_data.resample('10min').mean().reset_index()
    out = out.fillna(0)
    out['time'] = out.date.dt.hour.astype(str) + ":" + out.date.dt.minute.astype(str)
    out_compact = [out[['time', 'cars_up']], out[['time', 'cars_down']], out[['time', 'avg_speed_up']], out[['time', 'avg_speed_down']]]
    data = [[x.columns.values.tolist()] for x in out_compact]
    for d,o in zip(data,out_compact):
        for _,row in o.iterrows():
            d.append(row.values.tolist())
    return data, camera['loc_name'].iloc[0], camera['direction'].iloc[0], camera['lat'].iloc[0], camera['long'].iloc[0]

    

def getStatistics2(cam_code):
    camera = df_cameras[df_cameras['cam_code']==cam_code]
    df_data = pd.read_csv(data_path + 'data_collection_old.csv')
    df_data['date'] = pd.to_datetime(df_data['date'],format='%Y-%m-%d-%H-%M-%S')
    df_data = df_data.set_index('date')
    out = df_data.resample('30min').sum().reset_index()
    out['time'] = out.date.dt.hour.astype(str) + ":" + out.date.dt.minute.astype(str)
    out = out[['time', 'cars']]
    
    data = [out.columns.values.tolist()]
    for _, row in out.iterrows():
        data.append(row.values.tolist())
    print(camera)
    return data, camera['loc_name'].iloc[0], camera['direction'].iloc[0], camera['lat'].iloc[0], camera['long'].iloc[0]

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)