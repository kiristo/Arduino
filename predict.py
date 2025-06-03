import tensorflow as tf
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Assuming model.py is in the same directory and has load_model (though we reconstruct here)
# We don't strictly need model_utils.load_model if we are reconstructing, but good for consistency.

# Configuration
MODEL_PATH = 'address_parser_model.keras'
INPUT_TOKENIZER_PATH = 'input_tokenizer.pkl'
OUTPUT_TOKENIZER_PATH = 'output_tokenizer.pkl'
SEP_TOKEN = "<SEP>" # Should match train.py
START_TOKEN = "<START>" # Should match train.py
END_TOKEN = "<END>" # Should match train.py

OUTPUT_FIELDS = [
    'COMPLEMENT_DESTINATAIRE', 'NUMERO_VOIE', 'COMPLEMENT_NUMERO_VOIE',
    'TYPE_VOIE', 'LIBELLE_TYPE_VOIE', 'VOIE', 'LIEU_DIT',
    'COMPLEMENT_ADRESSE', 'CODE_POSTAL', 'COMMUNE'
]

def load_artefacts():
    """Loads tokenizers and the trained model."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, INPUT_TOKENIZER_PATH, OUTPUT_TOKENIZER_PATH]):
        print(f"Error: Model or tokenizer files not found. Please run train.py first.")
        return None, None, None

    with open(INPUT_TOKENIZER_PATH, 'rb') as f:
        input_tokenizer = pickle.load(f)
    with open(OUTPUT_TOKENIZER_PATH, 'rb') as f:
        output_tokenizer = pickle.load(f)

    trained_model = tf.keras.models.load_model(MODEL_PATH)
    return input_tokenizer, output_tokenizer, trained_model

def build_inference_models(trained_model):
    """Builds separate encoder and decoder models for inference."""
    # Encoder Model
    # Use the direct input tensor from the trained model's inputs list
    encoder_input_tensor = trained_model.input[0]
    # Retrieve layers by name, as defined in model.py
    encoder_embedding_layer = trained_model.get_layer('encoder_embedding')
    encoder_lstm_layer = trained_model.get_layer('encoder_lstm')

    # Build encoder model
    embedded_encoder_input = encoder_embedding_layer(encoder_input_tensor)
    _, state_h_enc, state_c_enc = encoder_lstm_layer(embedded_encoder_input)
    encoder_model = tf.keras.Model(encoder_input_tensor, [state_h_enc, state_c_enc], name="encoder_inference")

    # Decoder Model
    lstm_units = encoder_lstm_layer.units # LSTM units should be the same for encoder and decoder

    # Define new input layers for the decoder model for inference
    decoder_input_single_token = tf.keras.Input(shape=(1,), name='decoder_input_single_token')
    decoder_state_input_h = tf.keras.Input(shape=(lstm_units,), name='decoder_state_h_input')
    decoder_state_input_c = tf.keras.Input(shape=(lstm_units,), name='decoder_state_c_input')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding_layer = trained_model.get_layer('decoder_embedding')
    original_decoder_lstm_layer = trained_model.get_layer('decoder_lstm')
    decoder_dense_layer = trained_model.get_layer('decoder_output')

    # Create a new LSTM layer for inference, configured to return states
    # Copy weights from the trained layer
    inference_decoder_lstm = tf.keras.layers.LSTM(
        units=original_decoder_lstm_layer.units,
        return_sequences=True, # Must be true to get output at each step for Dense layer
        return_state=True,     # Must be true to get states for next step
        name='decoder_lstm_inference'
    )
    # Call the new LSTM layer once to build it, so we can set weights
    # Use dummy inputs that match expected shapes.
    # embedded_decoder_input will have shape (None, 1, embedding_dim)
    # decoder_states_inputs are [Input(shape=(lstm_units,)), Input(shape=(lstm_units,))]
    # We need a concrete embedding_dim.
    embedding_dim = decoder_embedding_layer.output_dim
    dummy_embedded_input = tf.keras.Input(shape=(1, embedding_dim), name="dummy_emb_input_for_build")
    dummy_states_input = [tf.keras.Input(shape=(lstm_units,), name="dummy_h_for_build"),
                          tf.keras.Input(shape=(lstm_units,), name="dummy_c_for_build")]

    # Build the layer by calling it
    _ = inference_decoder_lstm(dummy_embedded_input, initial_state=dummy_states_input)
    # Now set the weights
    inference_decoder_lstm.set_weights(original_decoder_lstm_layer.get_weights())


    embedded_decoder_input = decoder_embedding_layer(decoder_input_single_token)
    # Pass initial states to decoder LSTM
    # The LSTM layer will return three outputs now: sequences, state_h, state_c
    decoder_lstm_output_sequences, state_h_dec, state_c_dec = inference_decoder_lstm(
        embedded_decoder_input, initial_state=decoder_states_inputs
    )
    decoder_outputs = decoder_dense_layer(decoder_lstm_output_sequences) # Use the sequence output
    decoder_states = [state_h_dec, state_c_dec]

    decoder_model = tf.keras.Model(
        [decoder_input_single_token] + decoder_states_inputs,
        [decoder_outputs] + decoder_states,
        name="decoder_inference"
    )
    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, input_seq_padded, output_tokenizer,
                     input_max_len, output_loop_max_len):
    """Predicts the output sequence token by token."""
    states_value = encoder_model.predict(input_seq_padded, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index[START_TOKEN.lower()] # Ensure case consistency

    decoded_token_indices = []
    stop_condition = False

    for _ in range(output_loop_max_len):
        output_tokens_dist, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        sampled_token_index = np.argmax(output_tokens_dist[0, -1, :])

        if sampled_token_index == 0: # Should not happen if <UNK> or padding is 0 and vocab starts at 1
             # This might mean padding token, or an issue. For robust, check specific <END> token.
            print("Warning: Sampled token index is 0. This might be padding or an issue.")


        if sampled_token_index == output_tokenizer.word_index[END_TOKEN.lower()]: # Ensure case consistency
            stop_condition = True
            break

        decoded_token_indices.append(sampled_token_index)

        # Update the target sequence (of length 1)
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    # Convert indices to text
    # sequence_to_texts expects a list of sequences.
    if not decoded_token_indices:
        return "" # Or appropriate empty representation

    predicted_text = output_tokenizer.sequences_to_texts([decoded_token_indices])[0]
    return predicted_text


def format_prediction(predicted_text_sequence: str) -> list[str]:
    """Formats the raw predicted string into a list of fields."""
    # Tokenizer might already handle start/end, but good to be sure.
    # The output_tokenizer used filters='', so <start> and <end> are actual tokens.
    # The sequence_to_texts method might or might not include them based on tokenizer config.
    # Let's assume they might be there and remove them.

    # Remove special tokens if they are part of the text output
    # (depends on how tokenizer's sequence_to_texts handles them)
    # For now, let's assume they are not in the final text or handled by splitting.

    # Split by the separator token used during training target string creation
    # The target strings were like: "<START> field1 <SEP> field2 ... <END>"
    # After sequence_to_texts, it might be "field1 <sep> field2" (lowercase if tokenizer lowercases)
    # We need to ensure the SEP_TOKEN matches what the tokenizer outputs or what was used in training.
    # The output_tokenizer was created with filters='', so it should preserve case of tokens like <SEP>.

    # The predicted_text from tokenizer.sequences_to_texts([token_indices]) will be space-separated words.
    # e.g., "complement <sep> numero <sep> ..."
    # So we need to split by " " then reconstruct fields based on <sep>

    tokens = predicted_text_sequence.split(' ')
    if not tokens:
        return [""] * len(OUTPUT_FIELDS)

    structured_fields = []
    current_field = []
    sep_lower = SEP_TOKEN.lower() # train.py uses <SEP> which tokenizer might see as <sep>

    for token in tokens:
        if token == sep_lower:
            structured_fields.append(' '.join(current_field))
            current_field = []
        elif token == START_TOKEN.lower() or token == END_TOKEN.lower(): # Should not be here if loop is correct
            continue
        else:
            current_field.append(token)
    structured_fields.append(' '.join(current_field)) # Add the last field

    # Pad with empty strings if not enough fields were predicted
    while len(structured_fields) < len(OUTPUT_FIELDS):
        structured_fields.append("")
    # Truncate if too many fields were predicted
    return structured_fields[:len(OUTPUT_FIELDS)]


def main():
    print("Starting prediction script...")
    artefacts = load_artefacts()
    if artefacts[0] is None:
        return

    input_tokenizer, output_tokenizer, trained_model = artefacts

    print("Building inference models from trained model...")
    encoder_model, decoder_model = build_inference_models(trained_model)
    # encoder_model.summary() # Uncomment for debugging
    # decoder_model.summary() # Uncomment for debugging

    # Determine max lengths
    # Input max length from the shape of the encoder model's input layer
    input_max_len = encoder_model.input_shape[1]
    # Output max length for the prediction loop (can be generous)
    # This isn't strictly the output_max_len from training's padding, but how many steps we allow decoder to run.
    # A good estimate: number of fields * typical tokens per field + buffer
    output_loop_max_len = len(OUTPUT_FIELDS) * 5 + 10 # e.g., 10 fields, 5 tokens each + buffer
    # Or, can try to get from trained model's decoder input if it's fixed, but loop limit is safer.
    # output_sequence_len_from_training = trained_model.get_layer('decoder_input').input_shape[1]
    # if output_sequence_len_from_training:
    #     output_loop_max_len = output_sequence_len_from_training + 5 # Add a small buffer


    print(f"Determined Input Max Length: {input_max_len}")
    print(f"Using Output Loop Max Length: {output_loop_max_len}")


    test_addresses = [
        "8 RUE DUPORTAL 37000 TOURS",
        "123 MAIN ST APT 4B 12345 ANYTOWN",
        "LA MORLIERE 49370 LE LOUROUX BECONNAIS",
        "SERVICE DES ESPACES VERTS MAIRIE DE PARIS HOTEL DE VILLE 75004 PARIS"
    ]

    # Ensure start/end/sep tokens are in tokenizer, otherwise prediction loop will fail.
    required_tokens = [START_TOKEN.lower(), END_TOKEN.lower(), SEP_TOKEN.lower()]
    for token in required_tokens:
        if token not in output_tokenizer.word_index:
            print(f"CRITICAL ERROR: Token '{token}' not found in output_tokenizer.word_index.")
            print("This usually means train.py did not correctly include it during target string creation or tokenization.")
            print("Output Tokenizer word index (sample):")
            sample_idx = list(output_tokenizer.word_index.items())[:10]
            for k,v in sample_idx: print(f"  {k}: {v}")
            return

    for address_str in test_addresses:
        print(f"\nInput Address: '{address_str}'")

        # Preprocess input
        input_seq = input_tokenizer.texts_to_sequences([address_str.lower()]) # Assuming model trained on lowercase
        input_seq_padded = pad_sequences(input_seq, maxlen=input_max_len, padding='post', truncating='post')

        if not np.any(input_seq_padded): # Check if sequence is all zeros (all <UNK> or empty)
            print("  Warning: Input address resulted in an empty/unknown sequence after tokenization.")

        predicted_text = predict_sequence(
            encoder_model, decoder_model, input_seq_padded,
            output_tokenizer, input_max_len, output_loop_max_len
        )
        print(f"  Raw Predicted Text: '{predicted_text}'")

        structured_address_fields = format_prediction(predicted_text)

        print("  Predicted Structured Address:")
        for field_name, field_value in zip(OUTPUT_FIELDS, structured_address_fields):
            print(f"    {field_name}: {field_value.strip()}")

    print("\nPrediction script finished.")

if __name__ == '__main__':
    main()
