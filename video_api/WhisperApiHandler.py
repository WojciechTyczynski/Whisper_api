import requests


class WhisperApiHandler:
    def __init__(self, url):
        self.url = url

    def get_transcription(self, path) -> requests.Response:
        response = requests.post(f"{self.url}/transcribe/localfile/?localfile={path}")
        return response
