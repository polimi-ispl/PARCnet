import os
import yaml
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from parcnet import PARCnet
from metrics import nmse
from utils import simulate_packet_loss


def main():
    # ----------- Read config.yaml ----------- #

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Read paths from config file
    audio_test_path = Path(config['path']['audio_test_path'])
    model_checkpoint = Path(config['path']['nn_checkpoint_path'])
    trace_folder = Path(config['path']['trace_dir'])
    prediction_folder = Path(config['path']['prediction_dir'])

    # Read global params from config file
    sr = int(config['global']['sr'])
    packet_dim = int(config['global']['packet_dim'])
    extra_dim = int(config['global']['extra_pred_dim'])

    # Read AR params from config file
    ar_order = int(config['AR']['ar_order'])
    diagonal_load = float(config['AR']['diagonal_load'])
    num_valid_ar_packets = int(config['AR']['num_valid_ar_packets'])

    # Read NN params from config file
    num_valid_nn_packets = int(config['neural_net']['num_valid_nn_packets'])
    xfade_len_in = int(config['neural_net']['xfade_len_in'])

    print(str(audio_test_path))

    # ----------- Instantiate PARCnet ----------- #

    parcnet = PARCnet(
        packet_dim=packet_dim,
        extra_dim=extra_dim,
        ar_order=ar_order,
        ar_diagonal_load=diagonal_load,
        num_valid_ar_packets=num_valid_ar_packets,
        num_valid_nn_packets=num_valid_nn_packets,
        model_checkpoint=model_checkpoint,
        xfade_len_in=xfade_len_in,
        device='cpu'
    )

    # ----------- Load the reference audio file ----------- #

    y_ref, sr = librosa.load(audio_test_path, sr=sr, mono=True)

    # ----------- Load packet loss trace ----------- #

    trace_path = trace_folder.joinpath(audio_test_path.stem).with_suffix('.npy')
    trace = np.load(trace_path)

    # ----------- Simulate packet losses ----------- #

    y_lost = simulate_packet_loss(y_ref, trace, packet_dim)

    # ----------- Predict using PARCnet ----------- #

    y_pred = parcnet(y_lost, trace)

    # ----------- Save wav files ----------- #

    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)

    sf.write(prediction_folder.joinpath(f"{audio_test_path.stem}__zero-filling.wav"), y_lost.T, sr)
    sf.write(prediction_folder.joinpath(f"{audio_test_path.stem}__parcnet.wav"), y_pred.T, sr)

    # ----------- Compute NMSE ----------- #

    # Sample-wise mask, with 1s indicating missing samples and 0s indicating valid samples
    mask = np.repeat(trace, packet_dim).astype(bool)

    print('Zero-filling NMSE:')
    print(f'*** Signal: {nmse(y_lost, y_ref):.4f} dB')
    print(f'*** Packet: {nmse(y_lost[mask], y_ref[mask]):.4f} dB')

    print('PARCnet NMSE:')
    print(f'*** Signal: {nmse(y_pred, y_ref):.4f} dB')
    print(f'*** Packet: {nmse(y_pred[mask], y_ref[mask]):.4f} dB')


if __name__ == "__main__":
    main()
