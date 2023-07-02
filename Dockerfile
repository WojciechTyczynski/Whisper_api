FROM nvidia/cuda:12.2.0-base-ubuntu20.04
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
RUN apt-get install -y python3 python3-pip

# Change working directory
WORKDIR /code

# Copy requirements.txt
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./src /code/app

# 
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]