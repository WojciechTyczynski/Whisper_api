from typing import List, Optional

from pydantic import BaseModel


class VideoInput(BaseModel):
    video_url: str
    chunk_seconds: int = 60
    max_words_per_chunk: int = 120
    

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

class CustomSegment(BaseModel):
    start: float
    end: float
    text: str
