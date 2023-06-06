import loguru
import yt_dlp as youtube_dl
from fastapi import HTTPException

from models import *

# SHARED_FOLDER = "/home/mb/Whisper_api/shared"
SHARED_FOLDER = "/Users/wojtek/DTU/Thesis/Shared"
logger = loguru.logger


def download_youtube_audio(url: str):
    """
    Downloads the audio from a youtube video and saves it as a .wav file

    Parameters
    ----------
    url: str
        The url of the youtube video

    Returns
    -------
    str
        The name of the saved file
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": SHARED_FOLDER + "/%(id)s.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except youtube_dl.utils.DownloadError as e:
        logger.error("Error while downloading the youtube video: {}".format(e))
        raise HTTPException(status_code=400, detail="Error while downloading the youtube video")
    print("Downloaded the audio from the youtube video")
    # get the name of the downloaded file with wav extension
    filename = ydl.prepare_filename(ydl.extract_info(url, download=False))
    return filename


def concat_words_into_segments(
    whisper_transcript: Transcription, Video_data: VideoInput
):
    """
    Concatenate the transcript sections into chunks of maximum Seconds seconds

    Parameters
    ----------
    whisper_transcript: WhisperTranscription
        The transcript sections
    Video_data: VideoInput
        The maximum length of each transcript chunk and the video url
    Returns
    -------
    Transcription
        The concatenated transcript
    """
    logger.info(
        "Concatenating the transcript sections into chunks of maximum {} seconds".format(
            Video_data.chunk_seconds
        )
    )
    segments = []
    temp_segment = Segment(start=0, end=0, text="", words=[])
    index = 0
    # Create new segments of max video_data.chunk_seconds seconds and max video_data.max_words_per_chunk words
    for transcript_segment in whisper_transcript.segments:
        if (
            transcript_segment.end - index < Video_data.chunk_seconds
            and len(transcript_segment.words + transcript_segment.words)
            < Video_data.max_words_per_chunk
        ):
            temp_segment.end = transcript_segment.end
            temp_segment.text += transcript_segment.text
            temp_segment.words += transcript_segment.words
        else:
            # go over all words
            for word in transcript_segment.words:
                if (
                    word.end - index < Video_data.chunk_seconds
                    and len(temp_segment.words + [word])
                    < Video_data.max_words_per_chunk
                ):
                    temp_segment.end = word.end
                    temp_segment.text += word.word + " "
                    temp_segment.words += [word]
                else:
                    segments.append(temp_segment)
                    temp_segment = Segment(
                        start=word.start,
                        end=word.end,
                        text=word.word + " ",
                        words=[word],
                    )
                    index = word.end
    # add the last segment
    segments.append(temp_segment)
    return segments
