# Base image
FROM python:3.9

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

# Change working directory
WORKDIR /whisper_api

# Copy requirements.txt
COPY ./requirements.txt /whisper_api/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /whisper_api/requirements.txt

# 
COPY ./src /whisper_api/src

# 
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]