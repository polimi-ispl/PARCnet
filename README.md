# PARCnet: Hybrid Packet Loss Concealment for Real-Time Networked Music Applications

This repository will contain the accompanying code for 
> A. I. Mezza, M. Amerena, A. Bernardini, and A. Sarti, "Hybrid Packet Loss Concealment for Real-Time Networked Music Applications," in *IEEE Open Journal of Signal Processing*, 2023, doi: [10.1109/OJSP.2023.3343318](https://doi.org/10.1109/OJSP.2023.3343318).

## 🆕 Update (May 20, 2024) 
A new, updated version of PARCnet named **PARCnet-IS²** has been released! 

PARCnet-IS² was trained on 44.1 kHz single-instrument audio clips, works with packets of size 512 samples (11.6 ms), and features an improved inference mechanism that fixes cross-fading in case of a burst packet loss. 

PARCnet-IS² is the baseline model for the [IEEE-IS² 2024 Music Packet Loss Concealment Challenge](https://internetofsounds.net/ieee-is%C2%B2-2024-music-packet-loss-concealment-challenge/), which will be part of the **2nd IEEE International Workshop on Networked Immersive Audio** ([IEEE IWNIA 2024](https://internetofsounds.net/2nd-international-workshop-on-networked-immersive-audio/)), a satellite event of the **5th IEEE International Symposium on the Internet of Sounds** ([IEEE IS² 2024](https://internetofsounds.net/is2_2024/)).
 

Model weights, as well as tranining and inference code for PARCnet-IS² are available at the [official GitHub repo](https://github.com/polimi-ispl/2024-music-plc-challenge)! 

## Model Inference ✔️
In this repository, we provide all the necessary code to run a pretrained PARCnet model. 

----------------------

⚠️ **Note:** The inference code in this repo contains a known bug related to the cross-fade between consecutive missing packets. While we work on solving the issue, please check out [PARCnet-IS²](https://github.com/polimi-ispl/2024-music-plc-challenge), which correctly deals with burst losses.

----------------------

To test PARCnet using our piano example, simply run `example_parcnet_inference.py`. This will create two audio files in the `predictions` folder.

To test PARCnet using your own audio files, 
- Place your files in `test_data/audio`
- Update `audio_test_path` in the `config.yaml` file
- Run `create_default_trace.py` to create a trace in `test_data/traces/default`
- Run `example_parcnet_inference.py`

Make sure all test files are in __WAV format__.

## Model Training :warning:

----------------------

🔍 **Note:** Training code for **PARCnet-IS²** is now available. To train the original PARCnet model from scratch, please refer to the updated implementation available [here](https://github.com/polimi-ispl/2024-music-plc-challenge).

----------------------

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
Audio examples are available at our [GitHub demo page](https://polimi-ispl.github.io/PARCnet/).
