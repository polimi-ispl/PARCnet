# PARCnet: Hybrid Packet Loss Concealment for Real-Time Networked Music Applications

This repository will contain the accompanying code for 
> A. I. Mezza, M. Amerena, A. Bernardini and A. Sarti, "Hybrid Packet Loss Concealment for Real-Time Networked Music Applications," in *IEEE Open Journal of Signal Processing*, 2023, doi: [10.1109/OJSP.2023.3343318](https://doi.org/10.1109/OJSP.2023.3343318).

## Model Inference ✔️
In this repository, we provide all the necessary code to run a pretrained PARCnet model. 

To test PARCnet using our piano example, simply run `example_parcnet_inference.py`; this will create two audio files in the `predictions` folder.

To test PARCnet using your own audio files, 
- Upload your wav files in `test_data/audio`
- Update `audio_test_path` in the `config.yaml` file
- Run `create_default_trace.py` to create a trace in `test_data/traces/default`
- Run `example_parcnet_inference.py`

## Model Training :warning:

__The scripts for training a model from scratch will be made available soon.__

The PARCnet's neural branch provided in this repository was trained on 1000 tracks taken from the [MAESTRO Dataset V3.0.0](https://magenta.tensorflow.org/datasets/maestro), a large corpus of virtuoso piano recordings. The training hyperparameters are reported in `parcnet/config.yaml`. For further details, please refer to our [paper](https://doi.org/10.1109/OJSP.2023.3343318).

The current version of PARCnet was trained with
- __A sampling rate of 32 kHz.__ We encourage you not to change `sr` in `config.yaml`. 
- __Packets of 10 ms__ (320 samples at 32 kHz). We encourage you not to change `packet_dim` in `config.yaml`. 

Nevertheless, the code will not break if a different sampling rate or packet size is chosen. Whereas other inference scenarios have not been tested, it seems that PARCnet is somehow still able to work at 16 kHz with packets of 10 ms and, to some extent, 20 ms.

## Packet Traces 📦
To generalize the inference mechanism, we use *traces*. A trace is a `np.ndarray` containing a binary sequence of 1s and 0s. A `1` indicate a packet loss, whereas a `0` indicate that the corresponding packet was correctly received (valid packet). We provide a script to create *default* traces, i.e., traces with evenly-spaced losses; see `parcnet/create_default_trace.py`.

Traces depend on the chosen sampling frequency and packet size.

If you wish to modify `global` or `path` parameters in `config.yaml`, please run `create_default_trace.py` after the changes have been made.

## Audio Examples 🎧
Audio examples are available at our [GitHub page](https://polimi-ispl.github.io/PARCnet/).
