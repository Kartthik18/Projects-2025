# Digit Generation using DCGAN (MNIST)

This project trains a Deep Convolutional GAN (DCGAN) to generate handwritten digits from the **MNIST dataset**.

## Files
- `gan.py` — Generator, Discriminator, training loop
- `outputs/` — Generated images across epochs (example results included)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision matplotlib tqdm numpy
