from typing import BinaryIO, Union
import ffmpeg
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from loguru import logger
from omegaconf import OmegaConf
import whisper

# import utilities as ut
# from config import api_config


SAMPLE_RATE=16000
model = whisper.load_model('tiny')

def _load_audio_file(file: BinaryIO, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

app = FastAPI()

# Define health endpoint
@app.get('/health')
def health():
    return {'status': 'ok'}

# Define root endpoint
@app.get("/")
async def main():
    content = """
<body>
<form action="/transcribe/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

# @app.get('/greet', status_code=200)
# def say_hello():
#     return api_config.greet_message

# @app.get('/config', status_code=200)
# def get_config():
#     return {'config': OmegaConf.to_container(api_config)}

# Define endpoint to transcribe a file
@app.post("/transcribe/")
def transcribe_file(
    audio_files: list[UploadFile] = File()
):
    """
    Transcribe a list of audio/video files
    Parameters
    ----------
    audio_files: list[UploadFile]
        The audio/video file like objects
    Returns
    -------

    """
    response = {}
    for audio_file in audio_files:
        logger.info(f"File loaded: {audio_file.filename}")
        logger.info(f"Converting audio file...")
        audio = _load_audio_file(audio_file.file)
        logger.info(f"Audio file converted")
        logger.info(f"Transcribing audio file...")
        transcribtion = model.transcribe(audio)
        df = pd.DataFrame(transcribtion['segments'])
        results = df[['start', 'end', 'text']].to_dict('dict')
        # results = ut.load_audio_file(audio_file.file)
        response[audio_file.filename] = results
    return response

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)