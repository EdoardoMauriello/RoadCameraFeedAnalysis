# Road Camera Feed Analysis - Final Project README
## Project Overview
This project involves the analysis of road camera feeds obtained from aspi.it. The goal is to track vehicles in each frame, estimate their speeds and movements, and save this data into a textual dataset. This dataset will be useful for predicting traffic information and improving traffic management systems.

## Table of Contents
- Introduction
- Project Structure
- Dependencies
- Installation
- Usage
- Dataset
- Methodology
- Results
- Future Work
- Contributors
- License

## Dataset
ASPI, the italian national highway roads company provides a publicly available widespread webcam system updated every 5 minutes with 15 seconds of video: [aspi.it](https://www.autostrade.it/en/webcam). The first part of the project will focus on four webcams for training:
- Gazzada sud:      https://video.autostrade.it/video-mp4_hq/dt2/549646d9-d7d8-42bb-8306-b2f5ea21484e-1.mp4
- Castronno nord:   https://video.autostrade.it/video-mp4_hq/dt2/549646d9-d7d8-42bb-8306-b2f5ea21484e-1.mp4
- Solbiate sud:     https://video.autostrade.it/video-mp4_hq/dt2/f525e7d5-27f6-4435-bee4-c2b67f29ca73-0.mp4
- Cavaria sud:      https://video.autostrade.it/video-mp4_hq/dt2/6eb05a6d-23bb-49ab-afbe-21c3f212df80-0.mp4
Each URL is static for that webcam feed and the file is overwritten every about 5 minutes.

This project aims at training a model with the data obtained from the webcams and analyzing it in order to obtain traffic data over the course of some days and save it in a CSV for further analysis. 

## Usage
The repository provides a container that can be run with docker.

## How to run the app ##

```
docker build --tag python-docker .          # Build the image
docker run -it python-docker /bin/bash      # Debug docker build issues
docker run -d -p 8080:8080 python-docker    # Run the container
docker ps                                   # Check if the container is running
docker logs xxxxxxx                         # Logs in case of error
```
