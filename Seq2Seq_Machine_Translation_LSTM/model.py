from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

def build_seq2seq(english_vocab_size, french_vocab_size, embed_dim=200, lstm_units=256):
    # Encoder
    encoder_inputs = Input(shape=(None,), name="encoder_input")
    encoder_embed = Embedding(input_dim=english_vocab_size, output_dim=embed_dim)(encoder_inputs)
    _, state_h, state_c = LSTM(units=lstm_units, return_state=True, activation="relu")(encoder_embed)

    # Decoder
    decoder_inputs = Input(shape=(None, 1), name="decoder_input")
    decoder_lstm = LSTM(units=lstm_units, return_sequences=True, return_state=True, activation="relu")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    decoder_dense = Dense(french_vocab_size, activation="softmax", name="decoder_dense")
    outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    return model, encoder_inputs, decoder_inputs, state_h, state_c, decoder_lstm, decoder_dense
