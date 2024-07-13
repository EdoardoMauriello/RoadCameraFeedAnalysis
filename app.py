from flask import Flask,render_template
import pandas as pd

app = Flask(__name__)
data_path = 'images/'

@app.route('/')
def home():
    data = getStatistics()
    return render_template('index.html', data = data)

def getStatistics():
    df = pd.read_csv(data_path + 'data_collection.csv')
    df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d-%H-%M-%S')
    df = df.set_index('date')
    out = df.resample('30min').sum().reset_index()
    out['time'] = out.date.dt.hour.astype(str) + ":" + out.date.dt.minute.astype(str)
    out = out[['time', 'cars']]
    
    data = [out.columns.values.tolist()]
    for index, row in out.iterrows():
        data.append(row.values.tolist())
    return data

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)