from typing import List, Optional

from pydantic import BaseModel


# Define response model
class WordTimestamp(BaseModel):
    word: str
    tokens: List[int]
    start: float
    end: float


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: Optional[List[WordTimestamp]] = None


class Transcription(BaseModel):
    file: str
    segments: List[Segment]
    text: str
    language: str


class TranscriptionList(BaseModel):
    transcriptions: List[Transcription]


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
