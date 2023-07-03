FROM python:3.9
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

# Change working directory
WORKDIR /code

# Copy requirements.txt
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./src /code/app

# 
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]