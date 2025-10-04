# Seq2Seq Machine Translation with LSTM

This project implements a sequence-to-sequence (encoder–decoder) model using **LSTMs** to translate English → French on a toy dataset.

## Structure
- `preprocessing.py` — Tokenization, padding, vocabulary
- `model.py` — Encoder/Decoder LSTM model
- `train.py` — Training loop
- `inference.py` — Greedy decoding for translation
- `data/` — English/French toy dataset

## Setup
```bash
pip install tensorflow keras numpy
