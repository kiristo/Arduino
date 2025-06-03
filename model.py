import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import os

def build_model(input_vocab_size: int, output_vocab_size: int,
                input_max_len: int, output_max_len: int,
                embedding_dim: int = 128, lstm_units: int = 256):
    """
    Builds and compiles a sequence-to-sequence model.

    Args:
        input_vocab_size (int): Size of the input vocabulary.
        output_vocab_size (int): Size of the output vocabulary.
        input_max_len (int): Maximum length of input sequences.
        output_max_len (int): Maximum length of output sequences.
        embedding_dim (int, optional): Dimension of the embedding layer. Defaults to 128.
        lstm_units (int, optional): Number of units in the LSTM layers. Defaults to 256.

    Returns:
        tf.keras.models.Model: The compiled Keras model.
    """
    # Encoder
    encoder_inputs = Input(shape=(input_max_len,), name='encoder_input')
    encoder_embedding = Embedding(input_dim=input_vocab_size, output_dim=embedding_dim,
                                  name='encoder_embedding')(encoder_inputs)
    # We need return_state=True to get the hidden and cell states from the encoder LSTM
    encoder_lstm, state_h, state_c = LSTM(lstm_units, return_state=True, name='encoder_lstm')(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    # Decoder input will be the target sequence, shifted by one time step (teacher forcing)
    decoder_inputs = Input(shape=(output_max_len,), name='decoder_input')
    # The decoder embedding layer needs its own vocabulary size (output_vocab_size)
    decoder_embedding_layer = Embedding(input_dim=output_vocab_size, output_dim=embedding_dim,
                                     name='decoder_embedding')
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    # The decoder LSTM uses the encoder states as its initial state
    # return_sequences=True is needed because we want output at each time step
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=False, name='decoder_lstm')
    decoder_outputs = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Output layer
    # The Dense layer predicts the probability distribution over the output vocabulary
    decoder_dense = Dense(output_vocab_size, activation='softmax', name='decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    # The model takes encoder_inputs and decoder_inputs and outputs decoder_outputs
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq_model')

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) # 'accuracy' can be misleading for seq2seq, but good for a start

    return model

def save_model(model, file_path: str):
    """
    Saves the Keras model to the given file_path.

    Args:
        model (tf.keras.models.Model): The Keras model to save.
        file_path (str): The path where the model should be saved.
    """
    try:
        model.save(file_path)
        print(f"Model saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving model to {file_path}: {e}")

def load_model(file_path: str):
    """
    Loads a Keras model from the given file_path.

    Args:
        file_path (str): The path from where the model should be loaded.

    Returns:
        tf.keras.models.Model: The loaded Keras model, or None if loading fails.
    """
    try:
        model = tf.keras.models.load_model(file_path)
        print(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        return None

if __name__ == '__main__':
    # Define some dummy parameters
    INPUT_VOCAB_SIZE = 1000
    OUTPUT_VOCAB_SIZE = 1500
    INPUT_MAX_LEN = 20
    OUTPUT_MAX_LEN = 25
    EMBEDDING_DIM = 64  # Smaller for faster dummy test
    LSTM_UNITS = 128    # Smaller for faster dummy test

    print("Building a dummy model...")
    dummy_model = build_model(
        input_vocab_size=INPUT_VOCAB_SIZE,
        output_vocab_size=OUTPUT_VOCAB_SIZE,
        input_max_len=INPUT_MAX_LEN,
        output_max_len=OUTPUT_MAX_LEN,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS
    )

    print("\nOriginal Model Summary:")
    dummy_model.summary()

    temp_model_path = 'temp_dummy_model.keras'
    print(f"\nSaving model to {temp_model_path}...")
    save_model(dummy_model, temp_model_path)

    print(f"\nLoading model from {temp_model_path}...")
    loaded_dummy_model = load_model(temp_model_path)

    if loaded_dummy_model:
        print("\nLoaded Model Summary:")
        loaded_dummy_model.summary()
        # Basic check: compare number of layers or a specific layer's config if needed
        if len(dummy_model.layers) == len(loaded_dummy_model.layers):
            print("\nOriginal and loaded model have the same number of layers. Save/load seems OK.")
        else:
            print("\nMismatch in layer count between original and loaded model!")
    else:
        print("\nFailed to load the model, skipping loaded model summary.")

    # Clean up the temporary model file
    if os.path.exists(temp_model_path):
        try:
            os.remove(temp_model_path)
            print(f"\nSuccessfully removed temporary model file: {temp_model_path}")
        except Exception as e:
            print(f"\nError removing temporary model file {temp_model_path}: {e}")
    else:
        print(f"\nTemporary model file {temp_model_path} not found for cleanup.")

    print("\nScript finished.")
