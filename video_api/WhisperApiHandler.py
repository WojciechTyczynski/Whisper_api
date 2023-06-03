import requests


class WhisperApiHandler:
    def __init__(self, url):
        self.url = url

    def get_transcription(self, path, word_level_timestamps) -> requests.Response:
        response = requests.post(f"{self.url}/transcribe/localfile/?localfile={path}&word_timestamps={word_level_timestamps}")
        return response
