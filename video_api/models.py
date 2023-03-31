from pydantic import BaseModel
from typing import List

class VideoInput(BaseModel):
    video_url: str
    seconds: int = 60

class Segments(BaseModel):
    start : int
    end : int
    text : str
    tokens : list

class Transcription(BaseModel):
    file: str
    url : str
    segments : List[Segments]