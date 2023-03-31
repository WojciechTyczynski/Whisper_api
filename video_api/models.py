from typing import List

from pydantic import BaseModel


class VideoInput(BaseModel):
    video_url: str
    seconds: int = 60


class Segments(BaseModel):
    start: int
    end: int
    text: str
    tokens: list


class Transcription(BaseModel):
    file: str
    url: str
    segments: List[Segments]
