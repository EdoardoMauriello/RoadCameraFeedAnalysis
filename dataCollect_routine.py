from collections import defaultdict
import os
import cv2
import numpy as np
import pandas as pd
import csv
import requests
from datetime import datetime
from ultralytics import YOLO
import random

def collect(codCam, saveprocessedvideo = True, savelog = False, showprocessedvideo = False, folder = 'videos'):
    """
    Downloads the video and analyzes it with yolov8.

    Args:
        cameracode: Name of the input file.
        saveprocessedvideo: Flag for saving the processed video.
        savelog: Flag for saving the logfile.
        showprocessedvideo: Flag for showing using cv2.imshow() the processed video.
    """
    # Load the YOLOv8 model
    #model = YOLO("yolov8n.pt")
    model = YOLO("yolov8m-seg.pt")
    # Open the video file
    filename = f'{codCam[5:]}.mp4'
    print(filename)
    video_path = f'{folder}/{filename}'
    print(video_path)
    processed_video_path = f'{folder}/p{filename}'
    cap = cv2.VideoCapture(video_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video output
    if saveprocessedvideo:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    y1 = 140
    y2 = height - 40
    x1 = 0
    x2 = width

    # Store the track history with additional information
    track_history = defaultdict(lambda: {"points": [], "direction": None, "speed": 0, "sum_of_speeds": 0})

    # Store the statistics for the video
    track_stats = defaultdict(lambda: {"date": timestamp,"cars_up": 0,"cars_down": 0,"avg_speed_up": 0,"avg_speed_down": 0,"codCam": codCam})

    # Define CSV output file and headers
    csv_filename = f'r{codCam}.csv'
    csv_headers_collection = ["date", "cars_up", "cars_down", "avg_speed_up", "avg_speed_down", "codCam"]
    csv_headers_hist = ["frame", "track_id", "x", "y", "direction", "speed"]
    if savelog:
        with open(csv_filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers_hist)
            writer.writeheader()

    # Loop through the video frames
    frame_count = 0
    while cap.isOpened():
        # Read a frame from the video
        success, original_frame = cap.read()
        if not success:
            if frame_count == 0:
                print('Video not found')
            else:
                print('Video terminated')
            break

        frame_count += 1

        try:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(original_frame, persist=True, conf=0.2)
            try:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # Process and save track data to CSV
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]


                    cv2.circle(original_frame, (int(x),int(y)), 1, (0, 0, 255), -1)
                    if y < y1:
                        cv2.rectangle(original_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                        cv2.putText(original_frame, f"ID {track_id}", (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print("skipped")
                        continue
                    cv2.rectangle(original_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 153, 254), 2)
                    cv2.putText(original_frame, f"ID {track_id}", (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 153, 254), 2)

                    # Append center point and detect direction based on Y change
                    new_point = (float(x), float(y))
                    if len(track["points"]) > 0:
                        previous_point = track["points"][-1]
                        # Calculate distance travelled in pixels
                        distance = np.linalg.norm(np.array(new_point) - np.array(previous_point))
                        # Assuming constant frame rate (e.g., 30fps), estimate speed in pixels/frame
                        speed = distance if frame_count > 1 else 0  # Avoid division by zero
                        track["speed"] = speed
                        track["sum_of_speeds"] += speed
                        track["speed"] = distance if frame_count > 1 else 0  # Avoid division by zero
                        track_history[track_id]["direction"][0] = track_history[track_id]["direction"][0] + 1 if new_point[1] < previous_point[1] else track_history[track_id]["direction"][0] - 1
                        track_history[track_id]["direction"][1] = track_history[track_id]["direction"][1] + 1 if new_point[0] > previous_point[0] else track_history[track_id]["direction"][1] - 1
                    else:
                        track["speed"] = 0

                    track["points"].append(new_point)

                    track["points"].append(new_point)


                    print(f"Frame {frame_count}, Track ID {track_id}: {new_point}, Speed: {track['speed']} pixels/frame")

                    # Save track data to CSV
                    if savelog:
                        with open(csv_filename, "a", newline = '', ) as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=csv_headers_hist)
                            writer.writerow(
                                {
                                    "frame": frame_count,
                                    "track_id": track_id,
                                    "x": new_point[0],
                                    "y": new_point[1],
                                    "direction": track_history[track_id]["direction"],
                                    "speed": track["speed"]
                                }
                            )

                    # Keep track history limited
                    if len(track["points"]) > 30:
                        track["points"].pop(0)
            except:
                print('no cars found')
        except:
            print('div by zero error')



        cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)
        if saveprocessedvideo:
            out.write(original_frame)
        if showprocessedvideo:
            cv2.imshow("Video", original_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    count_up, count_down, avg_speed_up, avg_speed_down = calculate_stats(track_history)
    with open("data/datacollection.csv", "a", newline = '', ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers_collection)
        writer.writerow(
            {
                "timestamp": timestamp,
                "cars_up": count_up,
                "cars_down": count_down,
                "avg_speed_up": avg_speed_up,
                "avg_speed_down": avg_speed_down,
                "codCam": codCam
            }
        )
        # Get the boxes and track IDs


    # Release resources
    cap.release()
    if saveprocessedvideo:
        out.release()
        #if os.path.isfile(processed_video_path) and os.path.isfile(video_path):
        #        os.remove(video_path)
    cv2.destroyAllWindows()

def collectfake(codCam, saveprocessedvideo = True, savelog = False, showprocessedvideo = False, folder = 'videos'):
    """
    Downloads the video and analyzes it with yolov8.

    Args:
        cameracode: Name of the input file.
        saveprocessedvideo: Flag for saving the processed video.
        savelog: Flag for saving the logfile.
        showprocessedvideo: Flag for showing using cv2.imshow() the processed video.
    """
    # Load the YOLOv8 model
    #model = YOLO("yolov8n.pt")
    #model = YOLO("yolov8m-seg.pt")
    # Open the video file
    filename = f'{codCam[5:]}.mp4'
    print(filename)
    video_path = f'{folder}/{filename}'
    print(video_path)
    processed_video_path = f'{folder}/p{filename}'
    cap = cv2.VideoCapture(video_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video output
    if saveprocessedvideo:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    y1 = 140
    y2 = height - 40
    x1 = 0
    x2 = width

    # Store the track history with additional information
    track_history = defaultdict(lambda: {"points": [], "direction": None, "speed": 0, "sum_of_speeds": 0})

    # Store the statistics for the video
    track_stats = defaultdict(lambda: {"date": timestamp,"cars_up": 0,"cars_down": 0,"avg_speed_up": 0,"avg_speed_down": 0,"codCam": codCam})

    # Define CSV output file and headers
    csv_filename = f'r{codCam}.csv'
    csv_headers_collection = ["timestamp", "cars_up", "cars_down", "avg_speed_up", "avg_speed_down", "codCam"]
    csv_headers_hist = ["frame", "track_id", "x", "y", "direction", "speed"]
    if savelog:
        with open(csv_filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers_hist)
            writer.writeheader()

    # Loop through the video frames
    frame_count = 0
    while cap.isOpened():
        # Read a frame from the video
        success, original_frame = cap.read()
        if not success:
            if frame_count == 0:
                print('Video not found')
            else:
                print('Video terminated')
            break

        frame_count += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'DEMO'
        text_size, _ = cv2.getTextSize(text, font, 1, 2)
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(original_frame, text, (text_x, text_y), font, 1, (0, 255, 0), 2)
        if saveprocessedvideo:
            out.write(original_frame)
        if showprocessedvideo:
            cv2.imshow("Video", original_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    #count_up, count_down, avg_speed_up, avg_speed_down = calculate_stats(track_history)
    count_up = random.randint(0,20)
    count_down = random.randint(0,20)
    avg_speed_up = random.random()
    avg_speed_down = random.random()
    
    with open("data/datacollection.csv", "a", newline = '', ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers_collection)
        writer.writerow(
            {
                "timestamp": timestamp,
                "cars_up": count_up,
                "cars_down": count_down,
                "avg_speed_up": avg_speed_up,
                "avg_speed_down": avg_speed_down,
                "codCam": codCam
            }
        )
        # Get the boxes and track IDs


    # Release resources
    cap.release()
    if saveprocessedvideo:
        out.release()
        #if os.path.isfile(processed_video_path) and os.path.isfile(video_path):
        #        os.remove(video_path)
    cv2.destroyAllWindows()


def downloadfromsource(cam_code, urlsource = 'https://video.autostrade.it/video-mp4_hq', destination = 'videos'):
    if not os.path.exists(destination):
        os.makedirs(destination)
    response = requests.get(f'{urlsource}{cam_code}.mp4')
    filename = f'{destination}/{cam_code[5:]}.mp4'
    with open(filename, 'wb') as f:
        try:
            f.write(response.content)
            print(f'saved file in {filename}')
        except:
            print('error saving the file')

def calculate_stats(trackhistory):
    count_up = 0
    count_down = 0 
    sum_speed_up = 0
    for _,v in trackhistory.items():
        try:
            average_speed = v["sum_of_speeds"]/len(v["points"])
        except:
            average_speed = -1

        if v["direction"][0]>0:
            count_up += 1
            sum_speed_up += average_speed
        else:
            count_down += 1
            sum_speed_down += average_speed
    avg_speed_up = sum_speed_up/count_up
    avg_speed_down = sum_speed_down/count_down
    return (count_up, count_down, avg_speed_up, avg_speed_down)

df = pd.read_csv('data/cameras.csv')
for _,row in df.iterrows():
    if bool(row['active']):
        downloadfromsource(row['cam_code'])
for _,row in df.iterrows():
    if bool(row['active']):
        collectfake(row["cam_code"], saveprocessedvideo=True)


