import os
from typing import BinaryIO, List

import ffmpeg
import numpy as np
import uvicorn
import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from loguru import logger
from transformers import AutoProcessor, WhisperTokenizer, pipeline
from whisper.audio import load_audio
import pathlib
from models import *
from utils_timing import *
from utils_language import _detect_language, _convert_code_to_language, _convert_language_to_code

app = FastAPI()

SAMPLE_RATE = 16000
# SHARED_FOLDER = "/home/mb/Whisper_api/shared"
SHARED_FOLDER = pathlib.Path("/Users/wojtek/DTU/Thesis/Shared")
if not SHARED_FOLDER.exists():
    raise FileNotFoundError("Shared folder not found")
# sot_sequence = (50258, 50259, 50359)  # <|startoftranscript|><|en|><|transcribe|>
prepend_punctuations = "\"'“¿([{-"
append_punctuations = "\"'.。,，!！?？:：”)]}、"
model_prefix = "base"
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# Load whisper model
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=device,
)
model = pipe.model.to(device)
tokenizer = pipe.tokenizer
processor = AutoProcessor.from_pretrained("openai/whisper-base")

SPECIAL_TOKENS = {k:v for k, v in zip(tokenizer.all_special_tokens ,tokenizer.all_special_ids)}

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
async def transcribe_file(
    audio_files: List[UploadFile] = File(...),
    word_timestamps: bool = True,
    language: Optional[str] = None,
) -> TranscriptionList:
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
        try:
            audio = _load_audio_file(audio_file.file)
            logger.info("Audio file converted")
            if language is None:
                input_features = processor(audio, sampling_rate=16000,
                               return_tensors="pt").input_features.to(device)   
                languages_prob = _detect_language(device, model, tokenizer, input_features)[0]
                language_token = max(languages_prob, key=languages_prob.get)
                audio_language = _convert_code_to_language(language_token[2:-2])
            else:
                audio_language = language
                language_token = "<|" + _convert_language_to_code(audio_language) + "|>"
            logger.info(f'{"Language detected: "}{audio_language}')
            logger.info(f'{"Language token: "}{language_token}')
            tokenizer.set_prefix_tokens(language=audio_language)
            trans = _get_transcription(
                word_timestamps, audio=audio, file_name=audio_file.filename, language_token=language_token
            )
            logger.info("Audio file transcribed")
        except RuntimeError as e:
            logger.error(f"Failed to transcribe audio file: {e}")
            responses.append(
                Transcription(
                    file_name=audio_file.filename, text="", segments=[], language="",
                )
            )
            continue
        responses.append(trans)
    return TranscriptionList(transcriptions=responses)


def _get_transcription(word_timestamps, audio, file_name="", language_token=None):
    logger.info("Transcribing audio file...")
    output_pipeline = pipe(
        audio, return_timestamps=True, chunk_length_s=30, stride_length_s=[6,0], batch_size=16,
        generate_kwargs = {"task":"transcribe", "language":language_token, "no_repeat_ngram_size":5}
    )
    segments_output = []
    if word_timestamps:
        segments = get_segments(output_pipeline, audio, tokenizer)
        for key in segments.keys():
            text_tokens = segments[key]["tokens"]
            input_audio = processor(
                segments[key]["audio"], sampling_rate=16000, return_tensors="pt"
            )
            input_features = input_audio.input_features.to(device)
            sot_sequence = (50258, SPECIAL_TOKENS[language_token], 50359)  # <|startoftranscript|><|language|><|transcribe|>
            tokens = (
                torch.tensor(
                    [
                        *sot_sequence,
                        tokenizer.all_special_ids[-1],  # <|notimestamps|>
                        *text_tokens,
                        tokenizer.eos_token_id,
                    ]
                )
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                outputs = model(
                    input_features, decoder_input_ids=tokens, output_attentions=True,
                )
            cross_attentions = outputs.cross_attentions
            alignment_heads = get_alignment_heads(model_prefix, model)
            word_timestamps = find_alignment(
                cross_attentions,
                text_tokens,
                alignment_heads,
                tokenizer=tokenizer,
                segments_starts=[segments[key]["start"]],
            )
            segments_output.append(
                Segment(
                    start=segments[key]["start"],
                    end=segments[key]["end"],
                    text=segments[key]["text"],
                    words=word_timestamps[0],
                )
            )
    else:
        for chunk in output_pipeline["chunks"]:
            segments_output.append(
                Segment(
                    start=chunk["timestamp"][0],
                    end=chunk["timestamp"][1],
                    text=chunk["text"],
                )
            )

    trans = Transcription(
        file=file_name,
        segments=segments_output,
        text=output_pipeline["text"],
        language=language_token[2:-2],
    )
    return trans


# gets path to local file and returns transcription
@app.post("/transcribe/localfile/")
async def transcribe_local_file(
    localfile: str,
    word_timestamps: bool = True,
    language: Optional[str] = None,
) -> Transcription:
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

    print(f"File loaded: {path}")
    # load audio file
    audio = load_audio(path)
    try:
        if language is None:
            input_features = processor(audio, sampling_rate=16000,
                               return_tensors="pt").input_features.to(device)   
            languages_prob = _detect_language(device, model, tokenizer, input_features)[0]
            language_token = max(languages_prob, key=languages_prob.get)
            audio_language = _convert_code_to_language(language_token[2:-2])
        else:
            audio_language = language
            language_token = "<|" + _convert_language_to_code(audio_language) + "|>"
        tokenizer.set_prefix_tokens(language=audio_language)
        trans = _get_transcription(word_timestamps, audio=audio, file_name=localfile, language_token=language_token)
        logger.info(f'{"Language detected: "}{audio_language}')
    except Exception as e:
        logger.error(f"Failed to transcribe audio file: {e}")
        raise HTTPException(status_code=400, detail="Failed to transcribe file")
    return trans


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
