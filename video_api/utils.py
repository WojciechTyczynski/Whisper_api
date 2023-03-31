import yt_dlp as youtube_dl

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
        'format': 'bestaudio/best',
        'outtmpl': SHARED_FOLDER + '/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # get the name of the downloaded file with wav extension
    filename = ydl.prepare_filename(ydl.extract_info(url, download=False))
    return filename