import os
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import download_youtube_audio, concat_words_into_segments
from WhisperApiHandler import WhisperApiHandler

from models import *

app = FastAPI()

whisper_api = WhisperApiHandler("http://localhost:8000")

# Define health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/Video/transcribe")
def my_endpoint(Video_data: VideoInput) -> Transcription:
    """
    Takes a link to a video from the user and returns a transcript of the video.
    Transcripts are concatenated into chunks of maximum Seconds seconds.

    Only youtube videos are supported for now.

    Parameters
    ----------
    Video_data: VideoInput
        The video url and the maximum length of each transcript chunk

    Returns
    -------
    str
    """
    #  --- WE ONLY SUPPORT YOUTUBE VIDEOS FOR NOW ---
    # check if the url is a youtube video
    youtube_regex = r"^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$"
    if not re.match(youtube_regex, Video_data.video_url):
        raise HTTPException(status_code=400, detail="Please enter a valid youtube url")

    try:
        path = download_youtube_audio(Video_data.video_url)
    except:
        raise HTTPException(status_code=400, detail="Could not download the video")

    path = path.split(".")[0] + ".wav"
    filename = os.path.basename(path)
    # get the transcription
    response = whisper_api.get_transcription(filename, True)
    if response.status_code != 200:
        os.remove(path)
        raise HTTPException(status_code=400, detail="Could not transcribe the video")

    # remove the file from the shared folder
    os.remove(path)
    
    whisper_transcript = Transcription(**response.json())



    # concatenate the word level timestamps into chunks of maximum Seconds or maximum words
    segmented = concat_words_into_segments(whisper_transcript, Video_data)

    return segmented






if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
