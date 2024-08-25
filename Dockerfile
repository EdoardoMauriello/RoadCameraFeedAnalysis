# docker build --tag python-docker .  		# Build the image
## docker run -it python-docker /bin/bash   # Debug issues
# docker run -d -p 8080:8080 python-docker 	# Run the container
# docker ps                                 # Check if the container is running
# docker logs xxxxxxx                       # Logs in case of error

#### GCP ####
# gcloud init
# gcloud config set project uninsubria
# gcloud run deploy
## gcloud auth list


FROM python:3.8-slim-buster
WORKDIR /python-docker
COPY . /python-docker/
RUN apt-get update && apt-get install -y cron ffmpeg libsm6 libxext6
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Copy hello-cron file to the cron.d directory
COPY cron_file /etc/cron.d/cron_file
# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/cron_file
# Apply cron job
RUN crontab /etc/cron.d/cron_file 
# Create the log file to be able to run tail
RUN touch /python-docker/cron.log
CMD [ "python", "app.py"]
