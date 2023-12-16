import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


def _spectrogram(x: Tensor, fft_size: int, win_size: int, hop_size: int) -> Tensor:
    x_stft = torch.stft(
        input=x,
        n_fft=fft_size,
        win_length=win_size,
        hop_length=hop_size,
        return_complex=True,
    )

    return torch.clamp(x_stft.abs(), min=1e-7)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module.
    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        :param y_pred: positive magnitude STFT of the true signal
        :param y_true: positive magnitude STFT of the predicted signal
        :return: spectral convergence loss
        """
        return torch.norm(y_true - y_pred, p="fro") / torch.norm(y_true, p="fro")


class LogMagnitudeSTFTLoss(nn.Module):
    """Regularized log-magnitude loss module."""
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        :param y_pred: positive magnitude STFT of the true signal
        :param y_true: positive magnitude STFT of the predicted signal
        :return: log-magnitude L1-loss
        """
        return torch.mean(torch.abs(torch.log(y_true) - torch.log(y_pred)))


class SingleResolutionSTFTLoss(nn.Module):

    def __init__(self, fft_size: int, win_len: int, hop_size: int):
        super().__init__()
        self.fft_size = fft_size
        self.win_len = win_len
        self.hop_size = hop_size
        self.spec_conv_loss = SpectralConvergenceLoss()
        self.log_stft_loss = LogMagnitudeSTFTLoss()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param y_pred: predicted signal
        :param y_true: ground-truth signal
        :return: tuple of torch Tensors:
               - sc_loss - spectral convergence loss
               - log_loss - regularized log-magnitude STFT loss
        """

        stft_pred = _spectrogram(y_pred, self.fft_size, self.win_len, self.hop_size)
        stft_true = _spectrogram(y_true, self.fft_size, self.win_len, self.hop_size)

        sc_loss = self.spec_conv_loss(stft_pred, stft_true)
        log_loss = self.log_stft_loss(stft_pred, stft_true)

        return sc_loss, log_loss


class MultiResolutionSTFTLoss(nn.Module):

    def __init__(self,
                 fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
                 win_lengths: Tuple[int, ...] = (240, 600, 1200),
                 hop_sizes: Tuple[int, ...] = (50, 120, 240),
                 ):
        super().__init__()

        assert len(fft_sizes) == len(win_lengths) == len(hop_sizes), "Multi-resolution params must have the same length"

        self.R = len(fft_sizes)

        self.loss_functions = torch.nn.ModuleList()

        for n_fft, wlen, hop in zip(fft_sizes, win_lengths, hop_sizes):
            self.loss_functions.append(
                SingleResolutionSTFTLoss(n_fft, wlen, hop)
            )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param y_pred: predicted signal
        :param y_true: ground-truth signal
        :return: tuple of torch Tensors:
               - sc_loss - average multi-resolution Spectral Convergence loss
               - log_loss - average multi-resolution regularized log-magnitude STFT loss
        """
        sc_loss, log_loss = 0., 0.
        for fn in self.loss_functions:
            sc, log = fn(y_pred, y_true)
            sc_loss += sc
            log_loss += log
        sc_loss /= self.R
        log_loss /= self.R
        return sc_loss, log_loss
