import requests
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

print(os.path.dirname(os.path.abspath(__file__)))

data_path = '/python-docker/images/'
dcf = os.path.isfile( data_path + 'data_collection.csv')
if dcf == False :
    df = pd.DataFrame(columns=['date', 'cars', 'filename'])
    df.to_csv(data_path + 'data_collection.csv', index=False)


##### DATA COLLECTION 
now = datetime.now()
i_date =  now.strftime('%Y-%m-%d-%H-%M-%S')
img_url = 'https://www.astrogeo.va.it/data/webcam/current/ganna.jpg'
img_name = data_path + i_date + '.jpg'
img_data = requests.get(img_url).content

with open(img_name, 'wb') as handler:
    handler.write(img_data)
    
##### DATA PROCESSING
model = YOLO('yolov8n-seg.pt')
df = pd.read_csv(data_path + 'data_collection.csv')
results = model([img_name], conf=0.5, iou=0.6) 
for i, r in enumerate(results):
    img = r.plot()
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.xticks([]), plt.yticks([])
    plt.show()
    cars = 0
    for b in r.boxes:
        detection = model.names[int(b.data[0][-1])]
        if detection == 'car':
            cars = cars + 1
    
df.loc[df.size] = [i_date, cars, img_name]
df.to_csv(data_path + 'data_collection.csv', index=False)

##### DELETE IMAGE
os.remove(img_name)