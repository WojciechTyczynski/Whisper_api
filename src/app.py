import os
from typing import BinaryIO, List

import ffmpeg
import numpy as np
import uvicorn
import whisper
from transformers import pipeline, AutoProcessor
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from loguru import logger

from models import *
from utils_timing import *

app = FastAPI()

SAMPLE_RATE = 16000
SHARED_FOLDER = "E:\Whisper\Whisper_api\shared"
sot_sequence = (50258, 50259, 50359) # <|startoftranscript|><|en|><|transcribe|>
prepend_punctuations = "\"'“¿([{-"
append_punctuations = "\"'.。,，!！?？:：”)]}、"
model_prefix = "base"

# Load whisper model
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")
model = pipe.model
tokenizer = pipe.tokenizer
processor = AutoProcessor.from_pretrained("openai/whisper-base")

def _load_audio_file(file: BinaryIO, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py
    to accept a file object
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
            .run(
                cmd="ffmpeg",
                capture_stdout=True,
                capture_stderr=True,
                input=file.read(),
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


# Define health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


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
async def transcribe_file(audio_files: List[UploadFile] = File(...), word_timestamps: bool = True):
    """
    Transcribe a list of audio/video files
    Parameters
    ----------
    audio_files: list[UploadFile]
        The audio/video file like objects
    Returns
    -------
    A list of Transcription objects containing the file name,
    transcribed text, segments, and language
    """
    responses = []
    for audio_file in audio_files:
        logger.info(f'{"File loaded: "}{audio_file.filename}')
        logger.info("Converting audio file...")
        audio = _load_audio_file(audio_file.file)
        logger.info("Audio file converted")
        logger.info("Transcribing audio file...")
        # transcribtion = model.transcribe(audio)
        output_pipeline = pipe(audio, return_timestamps=True, chunk_length_s=30, batch_size=16)
        segments_output = []
        if word_timestamps:
            segments = get_segments(output_pipeline, audio)
            for key in segments.keys():
                text_tokens = tokenizer.encode(segments[key]['text'] , add_special_tokens=False)
                input_audio = processor(segments[key]['audio'], sampling_rate=16000, return_tensors="pt")
                input_features = input_audio.input_features  
                tokens = torch.tensor(
                [
                    *sot_sequence,
                    tokenizer.all_special_ids[-1],  # <|notimestamps|>
                    *text_tokens,
                    tokenizer.eos_token_id,
                ]
                ).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(
                        input_features, 
                        decoder_input_ids=tokens,
                        output_attentions=True,
                    )
                cross_attentions = outputs.cross_attentions
                alignment_heads = get_alignment_heads(model_prefix, model)
                word_timestamps = find_alignment(cross_attentions, text_tokens, alignment_heads, tokenizer=tokenizer, segments_starts=[segments[key]['start']])
                merge_punctuations(word_timestamps, prepend_punctuations,  append_punctuations)
                segments_output.append(
                    Segment(
                        start=segments[key]['start'],
                        end=segments[key]['end'],
                        text=segments[key]['text'],
                        words=word_timestamps[0],
                    )
                )
        else:
            for chunk in output_pipeline["chunks"]:
                segments_output.append(
                    Segment(
                        start=chunk['timestamp'][0],
                        end=chunk['timestamp'][1],
                        text=chunk["text"],
                    )
                )

        responses.append(
            Transcription(
                file=audio_file.filename,
                segments=segments_output,
                text=output_pipeline["text"],
                language='en',
            )
        )
    return TranscriptionList(transcriptions=responses)


# gets path to local file and returns transcription
@app.post("/transcribe/localfile/")
async def transcribe_local_file(localfile: str) -> WhisperTranscription:
    """
    Transcribe an audio file saved on shared drive
    Parameters
    ----------
    localfile: str
        The path to the audio file
    Returns
    -------
    A whisper transcription object
    """
    path = os.path.join(SHARED_FOLDER, localfile)

    # check if file exists
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    


    output_pipeline = pipe(path, return_timestamps=True, chunk_length_s=30, batch_size=16)
    response = Transcription(
        file=localfile,
        segments=output_pipeline["chunks"],
        text=output_pipeline["text"],
        language='en',
    )
    return response


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
