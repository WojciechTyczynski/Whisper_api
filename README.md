# Whisper API
This repository has 2 API's. The first is the ASR API, that loads a Huggingface Whisper model and gives the possibilty to transcribe an audio file. 
The other is a Connector API, which can download the audio from a youtube video and get the transcription by using the ASR Api.

The purpose if this is that the 2 API's are running at the same time and is sharing a common drive wheret the downloaded auido is stored. 
### GPU support 
The ASR API docker file is setup with a cuda enabled image. It is therefore necessary that the host device has an NVIDIA driver installed and preferably also a GPU. The docker image is using cuda version 12.0.0, and a higher or equal version is needed to be installed on the host device. 

### Running the API's
To run the API a docker compose file is given. The two API's have to be started at the same time. They share a common drive, which is also setup using the Docker Compose file. 

First the Images have to build. This can be done with: 
```
docker compose build
```
When the two Images have been build it can be run with 
```
docker compose up
```

