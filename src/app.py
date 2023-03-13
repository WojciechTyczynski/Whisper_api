from typing import BinaryIO, Union
import ffmpeg
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from loguru import logger

import whisper
from pydantic import BaseModel

# import utilities as ut
# from config import api_config

app = FastAPI()

SAMPLE_RATE=16000

# Load whisper model
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


# Define response model
class Transcription(BaseModel):
    file: str
    segments: list
    text: str
    language: str

class TranscriptionList(BaseModel):
    transcriptions: list[Transcription]


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


# Define endpoint to transcribe a file
@app.post("/transcribe/")
async def transcribe_file(
    audio_files: list[UploadFile] = File(...)
):
    """
    Transcribe a list of audio/video files
    Parameters
    ----------
    audio_files: list[UploadFile]
        The audio/video file like objects
    Returns
    -------
    A list of Transcription objects containing the file name, transcribed text, segments, and language
    """
    responses = []
    for audio_file in audio_files:
        logger.info(f"File loaded: {audio_file.filename}")
        logger.info(f"Converting audio file...")
        audio = _load_audio_file(audio_file.file)
        logger.info(f"Audio file converted")
        logger.info(f"Transcribing audio file...")
        transcribtion = model.transcribe(audio)
        responses.append(Transcription(
            file=audio_file.filename,
            segments=transcribtion['segments'],
            text=transcribtion['text'],
            language=transcribtion['language']
        ))
    return TranscriptionList(transcriptions=responses)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, workers=4)