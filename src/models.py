from typing import List

from pydantic import BaseModel


# Define response model
class Transcription(BaseModel):
    file: str
    segments: list
    text: str
    language: str


class TranscriptionList(BaseModel):
    transcriptions: List[Transcription]


class Segments(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class WhisperTranscription(BaseModel):
    text: str
    segments: List[Segments]
    language: str
