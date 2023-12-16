import os
import math
import yaml
import librosa
import argparse
import numpy as np
from pathlib import Path


def create_trace(loss_rate: int) -> None:
    # ----------- Read config.yaml ----------- #

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Read params from config file
    audio_test_path = Path(config['path']['audio_test_path'])
    trace_folder = Path(config['path']['trace_dir'])
    packet_dim = int(config['global']['packet_dim'])
    sr = int(config['global']['sr'])

    # load the clean signal
    y_true, sr = librosa.load(audio_test_path, sr=sr, mono=True)

    # ----------- Simulate packet losses ----------- #

    # Define the trace of lost packets: 1s indicate a loss
    trace_len = math.ceil(len(y_true) // packet_dim)
    trace = np.zeros(trace_len, dtype=int)
    trace[np.arange(loss_rate, trace_len, loss_rate)] = 1

    # ----------- Save trace ----------- #

    if not os.path.exists(trace_folder):
        os.makedirs(trace_folder)

    np.save(trace_folder.joinpath(f'{audio_test_path.stem}.npy'), trace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-r', '--loss_rate', type=int, default=10)
    args = vars(parser.parse_args())
    create_trace(args['loss_rate'])