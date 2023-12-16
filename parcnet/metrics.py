import librosa
import numpy as np
from torch import Tensor
from typing import Union


def _melspectrogram(x: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        x,
        n_mels=64,
        sr=32000,
        n_fft=512,
        power=1,
        hop_length=256,
        win_length=512,
        window='hann'
    )


def mel_spectral_convergence(y_pred: Union[Tensor, np.ndarray], y_true: Union[Tensor, np.ndarray]) -> np.ndarray:
    assert type(y_pred) == type(
        y_true), f'y_pred and y_true must be of the same type. Found {type(y_pred)} (y_pred) and {type(y_true)} (y_true).'

    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

    mel_true = _melspectrogram(y_true)
    mel_pred = _melspectrogram(y_pred)

    mel_sc = np.linalg.norm(mel_pred - mel_true, ord="fro") / np.linalg.norm(mel_true, ord="fro")

    return mel_sc


def nmse(y_pred: Union[Tensor, np.ndarray], y_true: Union[Tensor, np.ndarray]) -> np.ndarray:
    assert type(y_pred) == type(
        y_true), f'y_pred and y_true must be of the same type. Found {type(y_pred)} (y_pred) and {type(y_true)} (y_true).'

    if isinstance(y_pred, Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

    nmse_db = 20 * np.log10(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))

    return nmse_db
