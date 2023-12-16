import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from ar_model import ARModel
from model import HybridModel


class PARCnet:

    def __init__(self,
                 packet_dim: int,
                 extra_dim: int,
                 ar_order: int,
                 ar_diagonal_load: float,
                 num_valid_ar_packets: int,
                 num_valid_nn_packets: int,
                 model_checkpoint: str,
                 xfade_len_in: int,
                 device: str = 'cpu',
                 ):

        self.packet_dim = packet_dim

        # Define the prediction length, including the extra length
        self.pred_dim = packet_dim + extra_dim

        # Define the AR and neural network contexts in sample
        self.ar_context_len = num_valid_ar_packets * packet_dim
        self.nn_context_len = num_valid_nn_packets * packet_dim

        # Define fade-in modulation vector (neural network contribution only)
        self.fade_in = np.ones(self.pred_dim)
        self.fade_in[:xfade_len_in] = np.linspace(0, 1, xfade_len_in)

        # Define fade-out modulation vector
        self.fade_out = np.ones(self.pred_dim)
        self.fade_out[-extra_dim:] = np.linspace(1, 0, extra_dim)

        # Instantiate the linear predictor
        self.ar_model = ARModel(ar_order, ar_diagonal_load)

        # Load the pretrained neural network
        self.neural_net = HybridModel.load_from_checkpoint(model_checkpoint, channels=1, lite=True).to(device)

    def __call__(self, input_signal: np.ndarray, trace: np.ndarray, **kwargs) -> np.ndarray:
        self.neural_net.eval()
        output_signal = deepcopy(input_signal)

        for i, loss in tqdm(enumerate(trace), total=len(trace)):
            if loss:
                # Start index of the ith packet
                idx = i * self.packet_dim

                # AR model context
                valid_ar_packets = output_signal[idx - self.ar_context_len:idx]

                # AR model inference
                ar_pred = self.ar_model.predict(valid=valid_ar_packets, steps=self.pred_dim)

                # NN model context
                nn_context = output_signal[idx - self.nn_context_len: idx]
                nn_context = np.pad(nn_context, (0, self.pred_dim))
                nn_context = torch.tensor(nn_context[None, None, ...])

                with torch.no_grad():
                    # NN model inference
                    nn_pred = self.neural_net(nn_context)
                    nn_pred = nn_pred[..., -self.pred_dim:]
                    nn_pred = nn_pred.squeeze().cpu().numpy()

                # Apply fade-in to the neural network contribution (inbound fade-in)
                nn_pred *= self.fade_in

                # Combine the two predictions
                prediction = ar_pred + nn_pred

                # Cross-fade the compound prediction (outbound fade-out)
                prediction *= self.fade_out

                # Cross-fade the output signal (outbound fade-in)
                output_signal[idx:idx + self.pred_dim] *= 1 - self.fade_out

                # Conceal lost packet
                output_signal[idx: idx + self.pred_dim] += prediction

        return output_signal
