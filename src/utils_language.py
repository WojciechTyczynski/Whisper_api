from fastapi import HTTPException
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
from typing import List, Collection, Dict, Optional
import torch


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

LANGUAGES_CODES = {v: k for k, v in LANGUAGES.items()}


def _convert_code_to_language(code):
    """
    Convert language code to language name
    Parameters
    ----------
    code: str
        The language code
    Returns
    -------
    The language name
    """
    try:
        return LANGUAGES[code]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid language code")
    
def _convert_language_to_code(language):
    """
    Convert language name to language code
    Parameters
    ----------
    language: str
        The language name
    Returns
    -------
    The language code
    """
    try:
        return LANGUAGES_CODES[language]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid language name")
    

def _detect_language(device: str, model: WhisperForConditionalGeneration, tokenizer: WhisperTokenizer, input_features,
                    possible_languages: Optional[Collection[str]] = None) -> List[Dict[str, float]]:
    """
    Detect the language of each input
    Parameters
    ----------
    model: WhisperForConditionalGeneration
        The model
    tokenizer: WhisperTokenizer
        The tokenizer
    input_features: torch.Tensor
        The input features
    possible_languages: Optional[Collection[str]]
        The possible languages
    Returns
    -------
    A list of dictionaries mapping language to probability
    """
    # hacky, but all language tokens and only language tokens are 6 characters long
    language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
    if possible_languages is not None:
        language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]
        if len(language_tokens) < len(possible_languages):
            raise RuntimeError(f'Some languages in {possible_languages} did not have associated language tokens')

    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

    # 50258 is the token for transcribing
    logits = model(input_features,
                   decoder_input_ids = torch.tensor([[50258] for _ in range(input_features.shape[0])]).to(device)).logits
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float('inf')

    output_probs = logits.softmax(dim=-1).cpu()
    return [
        {
            lang: output_probs[input_idx, 0, token_id].item()
            for token_id, lang in zip(language_token_ids, language_tokens)
        }
        for input_idx in range(logits.shape[0])
    ]
