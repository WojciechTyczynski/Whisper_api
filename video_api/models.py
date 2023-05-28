from typing import List, Optional

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

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class WhisperSegments(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    words: Optional[List[WordTimestamp]] = None
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class WhisperTranscription(BaseModel):
    text: str
    segments: List[WhisperSegments]
    language: str
