from flask import Flask,render_template, request, url_for
import pandas as pd

app = Flask(__name__)
data_path = 'data/'
webcam_url_base = 'https://video.autostrade.it/video-mp4_hq'
df_cameras = pd.read_csv(data_path + 'cameras.csv')

@app.route('/')
def home():
    points = getMapPoints()
    return render_template('index.html', points = points)

@app.route('/dashboard')
def specific():
    camCode = request.args.get('camCode')
    data, locName, direction, lat, long = getStatistics(camCode)
    return render_template('camera_feed_dashboard.html', data = data, locName = locName, direction = direction, lat = lat, long = long, camURL = webcam_url_base + camCode)

def getMapPoints():
    points = [
    {"coords": [row["lat"], row["long"]], "url": request.url + 'dashboard?camCode=' + row["camCode"]}
    for _, row in df_cameras.iterrows()
    ]
    return points
    

def getStatistics(camCode):
    camera = df_cameras[df_cameras['camCode']==camCode]
    df_data = pd.read_csv(data_path + 'data_collection.csv')
    df_data['date'] = pd.to_datetime(df_data['date'],format='%Y-%m-%d-%H-%M-%S')
    df_data = df_data.set_index('date')
    out = df_data.resample('30min').sum().reset_index()
    out['time'] = out.date.dt.hour.astype(str) + ":" + out.date.dt.minute.astype(str)
    out = out[['time', 'cars']]
    
    data = [out.columns.values.tolist()]
    for _, row in out.iterrows():
        data.append(row.values.tolist())
    print(camera)
    return data, camera['locName'].iloc[0], camera['direction'].iloc[0], camera['lat'].iloc[0], camera['long'].iloc[0]

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)