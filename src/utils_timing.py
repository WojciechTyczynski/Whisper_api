import base64
import gzip

import numpy as np
import torch
from torch import nn

from models import *
def get_substringofrepeatedchars(string):
    """
    Function to get the substring of repeated characters in a string.
    """
    temp = string.split(" ")
    a = []
    for i in temp:
        if i not in a[:-10]:
            a.append(i)
    temp = " ".join(a)
    return temp

def get_segments(transcription, audio, tokenizer, sample_rate: int = 16000):
    """
    Function to merge the chunks of the transcription form Hugging Face WhisperPipeline into 
    <30s segments or <448 tokens. 
    """
    MAX_LEN_TOKENS = 448
    seek = 0
    current_seek = 0
    segments = {0: {"start": 0, "end": 0, "text": "", "tokens": []}}

    # because we are batching, then the last timestamp is missing. We add it here
    max_timestamp = len(audio) / sample_rate
    last_x, _ = transcription["chunks"][-1]["timestamp"]
    transcription["chunks"][-1]['timestamp'] = (last_x, max_timestamp)

    for i in range(len(transcription["chunks"])):
        seek_index = int(seek * 100)
        start, end = transcription["chunks"][i]["timestamp"]
        if i == 166:
            print(i)
        temp_tokens = tokenizer.encode(
            transcription["chunks"][i]["text"], add_special_tokens=False
        )
         # We see sometimes that the model do hallucinate and repeat the same sentence over and over
        # We try to detect this by looking at the length of the tokens and if it is > 448 we try to
        # get the substring of repeated characters in the sentence and encode it again.
        if len(temp_tokens) > MAX_LEN_TOKENS:
            transcription["chunks"][i]["text"] = get_substringofrepeatedchars(transcription["chunks"][i]["text"])
            temp_tokens = tokenizer.encode(transcription["chunks"][i]["text"], add_special_tokens=False)


        concat_len = len(segments[seek_index]["tokens"]) + len(temp_tokens)
        if end - seek < 29 and concat_len < MAX_LEN_TOKENS:
            segments[seek_index]["text"] = (
                segments[seek_index]["text"] + transcription["chunks"][i]["text"]
            )
            segments[seek_index]["tokens"] = (
                segments[seek_index]["tokens"] + temp_tokens
            )
            current_seek = end
        else:
            segments[seek_index]["end"] = current_seek
            segments[seek_index]["audio"] = audio[
                int(segments[seek_index]["start"] * sample_rate) : int(
                    segments[int(seek * 100)]["end"] * sample_rate
                )
            ]
            seek = current_seek
            segments[int(seek * 100)] = {
                "start": start,
                "text": transcription["chunks"][i]["text"],
                "tokens": temp_tokens,
            }
            current_seek = end
    segments[int(seek * 100)]["end"] = current_seek
    segments[int(seek * 100)]["audio"] = audio[
        int(segments[int(seek * 100)]["start"] * sample_rate) : int(
            segments[int(seek * 100)]["end"] * sample_rate
        )
    ]
    return segments


def get_alignment_heads(model_prefix, model):
    _ALIGNMENT_HEADS = {
        "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
        "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
        "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
        "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
        "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
        "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
        "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
        "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
        "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
        "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
        "large": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    }
    array = np.frombuffer(
        gzip.decompress(base64.b85decode(_ALIGNMENT_HEADS[model_prefix])), dtype=bool
    ).copy()
    mask = torch.from_numpy(array).reshape(
        model.config.decoder_layers, model.config.decoder_attention_heads
    )
    return mask.to_sparse().indices().T


import string
from typing import List


def split_tokens_on_unicode(tokens: List[int], tokenizer):
    decoded_full = tokenizer.decode(tokens, decode_with_timestamps=True)
    replacement_char = "\ufffd"

    words = []
    word_tokens = []
    current_tokens = []
    unicode_offset = 0

    for token in tokens:
        current_tokens.append(token)
        decoded = tokenizer.decode(current_tokens, decode_with_timestamps=True)

        if (
            replacement_char not in decoded
            or decoded_full[unicode_offset + decoded.index(replacement_char)]
            == replacement_char
        ):
            words.append(decoded)
            word_tokens.append(current_tokens)
            current_tokens = []
            unicode_offset += len(decoded)

    return words, word_tokens


def split_tokens_on_spaces(tokens: List[int], tokenizer):
    subwords, subword_tokens_list = split_tokens_on_unicode(tokens, tokenizer)
    words = []
    word_tokens = []

    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.eos_token_id
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or with_space or punctuation or len(words) == 0:
            words.append(subword)
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)

    return words, word_tokens


def median_filter(inputs: torch.Tensor, filter_width: int) -> torch.Tensor:
    """
    Applies a median filter of width `filter_width` along the last dimension of the input. 
    The `inputs` tensor is assumed to be 3- or 4-dimensional.
    """
    pad_width = filter_width // 2
    if inputs.shape[-1] <= pad_width:
        return inputs

    # Pad the left and right edges.
    inputs = nn.functional.pad(inputs, (pad_width, pad_width, 0, 0), mode="reflect")

    # sort() is faster than torch.median (https://github.com/pytorch/pytorch/issues/51450)
    result = inputs.unfold(-1, filter_width, 1).sort()[0][..., pad_width]
    return result


def dtw(x: np.ndarray):
    """
    Dynamic time warping. Used to generate token-level timestamps.
    """
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    # backtrace
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    text_indices = []
    time_indices = []
    while i > 0 or j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")
    text_indices = np.array(text_indices)[::-1]
    time_indices = np.array(time_indices)[::-1]
    return text_indices, time_indices


def find_alignment(
    cross_attentions,
    text_tokens,
    alignment_heads,
    tokenizer,
    batch_size: int = 1,
    segments_starts: List[int] = None,
    time_precision: float = 0.2,
    tokens_per_second: int = 50,
    num_frames: int = 3000,
):

    weights = torch.stack([cross_attentions[l][:, h, :, :] for l, h in alignment_heads])
    weights = weights.permute([1, 0, 2, 3])
    weights = weights[:, :, : num_frames // 2]
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = median_filter(weights, 7)

    # Average the different cross-attention heads.
    matrix = weights.mean(dim=1)
    matrix = matrix[:, 3:-1, :]  # Skip sot_sequence (50258, 50259, 50359)
    timings = []
    for i in range(batch_size):
        text_indices, time_indices = dtw(-matrix[i].double().cpu().numpy())
        words, word_tokens = split_tokens_on_spaces(
            text_tokens + [tokenizer.eos_token_id], tokenizer
        )
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] / tokens_per_second
        start_times = list(jump_times[word_boundaries[:-1]])
        end_times = list(jump_times[word_boundaries[1:]])
        try:
            merge_punctuations(words, word_tokens, start_times, end_times)
        except:
            raise RuntimeWarning("Failed to merge punctuations")
        timing = [
            WordTimestamp(
                word=word,
                tokens=tokens,
                start=start + segments_starts[i],
                end=end + segments_starts[i],
            )
            for word, tokens, start, end in zip(
                words, word_tokens, start_times, end_times
            )
        ]
        timings.append(timing)
    return timings


def merge_punctuations(
    words: List[str],
    tokens: List[List[int]],
    start_times: List[float],
    end_times: List[float],
    prepended: str = "\"'“¿([{-",
    appended: str = "\"'.。,，!！?？:：”)]}、",
):
    # merge prepended punctuations
    i = len(words) - 2
    j = len(words) - 1
    while i >= 0:
        previous = words[i]
        following = words[j]
        if previous.startswith(" ") and previous.strip() in prepended:
            # prepend it to the following word
            words[j] = words[i] + words[j]
            tokens[j] = tokens[i] + tokens[j]
            start_times[j] = start_times[i]
            words[i] = ""
            tokens[i] = []
            start_times[i] = -1
            end_times[i] = -1
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(words):
        previous = words[i]
        following = words[j]
        if not previous.endswith(" ") and following in appended:
            # append it to the previous word
            words[i] = words[i] + words[j]
            tokens[i] = tokens[i] + tokens[j]
            end_times[i] = end_times[j]
            words[j] = ""
            tokens[j] = []
            start_times[j] = -1
            end_times[j] = -1
        else:
            i = j
        j += 1

    # remove elements that are now empty
    words[:] = [word for word in words if word]
    tokens[:] = [token for token in tokens if token]
    start_times[:] = [idx for idx in start_times if idx != -1]
    end_times[:] = [idx for idx in end_times if idx != -1]
