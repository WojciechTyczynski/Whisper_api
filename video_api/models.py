from typing import List

from pydantic import BaseModel


class VideoInput(BaseModel):
    video_url: str
    seconds: int = 60


class Segment(BaseModel):
    start: int
    end: int
    text: str
    tokens: list


class Transcription(BaseModel):
    url: str
    segments: List[Segment]
    text: str
    language: str


class WhisperSegments(BaseModel):
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
    segments: List[WhisperSegments]
    language: str
