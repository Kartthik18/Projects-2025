# main.py
import math
import time
import random
import torch
import torch.nn as nn
from torch import optim

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from preprocessing import (
    SOS_token, EOS_token, MAX_LENGTH,
    tensorFromSentence,
    get_dataloader, prepareData
)
from model import TransformerModel, generate_square_subsequent_mask, create_padding_mask

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m:d}m {int(s):d}s'

# ---------------------------------------------------------------------
# Training (kept very close to your original)
# ---------------------------------------------------------------------
def train_transformer_epoch(dataloader, model, optimizer, criterion, teacher_forcing_ratio=1.0):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        # Move to device *here*
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # Shifted inputs/targets for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_pad_mask = create_padding_mask(src).to(device)
        tgt_pad_mask = create_padding_mask(tgt_input).to(device)
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)

        if random.random() < teacher_forcing_ratio:
            # Teacher forcing: gold target as input
            output = model(src, tgt_input, src_mask=src_pad_mask, tgt_mask=tgt_mask)
        else:
            # Greedy decode during training
            B, T_out = tgt_output.shape
            V = model.fc_out.out_features
            output = torch.zeros(B, T_out, V, device=device)
            decoder_input = torch.full((src.shape[0], 1), SOS_token, dtype=torch.long, device=device)
            for t in range(T_out):
                tgt_mask_step = generate_square_subsequent_mask(decoder_input.size(1), device=device)
                step_output = model(src, decoder_input, src_mask=src_pad_mask, tgt_mask=tgt_mask_step)
                next_token = step_output[:, -1, :].argmax(dim=-1, keepdim=True)
                output[:, t, :] = step_output[:, -1, :]
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_transformer(dataloader, model, n_epochs=10, lr=5e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD/SOS=0
    teacher_forcing_ratio = 0.5

    for epoch in range(1, n_epochs + 1):
        start = time.time()
        loss = train_transformer_epoch(
            dataloader,
            model,
            optimizer,
            criterion,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        print(f"[{asMinutes(time.time() - start)}] Epoch {epoch} - Loss: {loss:.4f}")

# ---------------------------------------------------------------------
# Inference (greedy)
# ---------------------------------------------------------------------
def evaluate_transformer(model, sentence, input_lang, output_lang, max_len=MAX_LENGTH):
    model.eval()
    with torch.no_grad():
        src = tensorFromSentence(input_lang, sentence).to(device)

        tgt_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
        src_mask = create_padding_mask(src).to(device)

        for _ in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)
            out = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            next_token = out[:, -1, :].argmax(-1).unsqueeze(1)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            if next_token.item() == EOS_token:
                break

        decoded_tokens = [
            output_lang.index2word[token.item()]
            for token in tgt_input.squeeze()
            if token.item() not in (SOS_token, EOS_token)
        ]
        return decoded_tokens

def evaluate_and_compare(model, input_lang, output_lang, pairs, n=5, show_bleu=True):
    model.eval()
    smoothie = SmoothingFunction().method4 if show_bleu else None

    for _ in range(n):
        pair = random.choice(pairs)
        src_sentence = pair[0]
        target_sentence = pair[1]

        prediction = evaluate_transformer(model, src_sentence, input_lang, output_lang)
        predicted_sentence = ' '.join(prediction)

        print("🟩 Input (French):        ", src_sentence)
        print("🟦 Ground Truth (English):", target_sentence)
        print("🟥 Prediction:            ", predicted_sentence)

        if show_bleu:
            reference = [target_sentence.split()]
            candidate = predicted_sentence.split()
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
            print(f"🔵 BLEU Score:             {bleu_score:.4f}")
        print()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Build dataloader & vocabs
    batch_size = 32
    input_lang, output_lang, pairs, train_dataloader = get_dataloader(batch_size=batch_size)

    # Init model
    transformer_model = TransformerModel(
        input_vocab_size=input_lang.n_words,
        target_vocab_size=output_lang.n_words,
        d_model=128,
        nhead=8,
        num_layers=5,
        dropout=0.1,
        max_len=MAX_LENGTH
    ).to(device)

    # Train (multiple phases like your original)
    train_transformer(train_dataloader, transformer_model, n_epochs=10, lr=5e-4)
    train_transformer(train_dataloader, transformer_model, n_epochs=5,  lr=5e-4)
    train_transformer(train_dataloader, transformer_model, n_epochs=5,  lr=5e-4)

    # Quick random eval like your printouts
    for _ in range(5):
        pair = random.choice(pairs)
        print("🟩 Input:        ", pair[0])
        print("🟦 Ground Truth: ", pair[1])
        pred = evaluate_transformer(transformer_model, pair[0], input_lang, output_lang)
        print("🟥 Prediction:   ", ' '.join(pred))
        print()

    # BLEU sampling
    evaluate_and_compare(transformer_model, input_lang, output_lang, pairs, n=5, show_bleu=True)
