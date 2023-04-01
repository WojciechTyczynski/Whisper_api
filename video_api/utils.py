import yt_dlp as youtube_dl

from models import *
SHARED_FOLDER = "E:\Whisper\Whisper_api\shared"


def download_youtube_audio(url):
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
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # get the name of the downloaded file with wav extension
    filename = ydl.prepare_filename(ydl.extract_info(url, download=False))
    return filename


def concat_sections_into_chunks(whisper_transcript: WhisperTranscription, Video_data: VideoInput) -> Transcription:
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

    
    
    final_transcript = Transcription(url=Video_data.video_url, segments=[], text=whisper_transcript.text, language=whisper_transcript.language) 

    temp_segment = Segment(
        start=whisper_transcript.segments[0].start,
        end=whisper_transcript.segments[0].end,
        text=whisper_transcript.segments[0].text,
        tokens=whisper_transcript.segments[0].tokens,
    ) 

    for segment in whisper_transcript.segments[1:]:
        if segment.end - temp_segment.start < Video_data.seconds:
            temp_segment.end = segment.end
            temp_segment.text += segment.text
            temp_segment.tokens += segment.tokens
        else:
            final_transcript.segments.append(temp_segment)
            temp_segment = segment.copy()

    final_transcript.segments.append(temp_segment)

    return final_transcript