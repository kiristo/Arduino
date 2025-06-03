import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Assuming data_loader.py and model.py are in the same directory
import data_loader
import model

# Configuration Parameters
DATA_FILE_PATH = '2025_100_Adresses.csv' # Will use dummy from data_loader if not present
MODEL_SAVE_PATH = 'address_parser_model.keras'
INPUT_TOKENIZER_PATH = 'input_tokenizer.pkl'
OUTPUT_TOKENIZER_PATH = 'output_tokenizer.pkl'
EPOCHS = 5 # Keep low for initial testing with dummy data
BATCH_SIZE = 8 # Smaller batch for tiny dummy dataset
LSTM_UNITS = 128 # Smaller for faster dummy test
EMBEDDING_DIM = 64 # Smaller for faster dummy test
SEP_TOKEN = "<SEP>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"

def create_target_strings(y_df: pd.DataFrame) -> list[str]:
    """Combines structured address fields into single target strings."""
    target_strings = []
    for _, row in y_df.iterrows():
        # Ensure all values are strings before joining
        fields = [str(field) for field in row.values]
        target_str = f"{START_TOKEN} {' '.join(fields)} {END_TOKEN}"
        # Replace multiple spaces that might arise from empty fields with a single space
        target_str = ' '.join(target_str.split())
        # The prompt asked for <SEP> token between fields, let's adjust
        # Current y_df has columns: 'COMPLEMENT_DESTINATAIRE', 'NUMERO_VOIE', etc.
        # Let's make it explicit with SEP
        processed_fields = []
        for field_name in y_df.columns:
            processed_fields.append(str(row[field_name]))

        target_str = f"{START_TOKEN} {f' {SEP_TOKEN} '.join(processed_fields)} {END_TOKEN}"
        target_strings.append(target_str)
    return target_strings

def main():
    print("Starting training script...")

    # --- 1. Load Data ---
    print(f"Loading data from {DATA_FILE_PATH}...")
    # Create dummy CSV if it doesn't exist (using data_loader's main block logic)
    if not os.path.exists(DATA_FILE_PATH):
        print(f"File {DATA_FILE_PATH} not found. Creating a dummy file for testing.")
        dummy_data = {
            'AdresseNonStructure': [
                '123 MAIN ST, ANYTOWN', '456 OAK AVE, SOMEWHERE', '789 PINE LN, ELSEWHERE',
                '101 ELM RD, NOWHERE', '202 MAPLE DR, ANYCITY', '303 BIRCH ST, TESTVILLE',
                '404 CEDAR AVE, SAMPLBURG', '505 WILLOW LN, MOCKCITY'
            ],
            'COMPLEMENT_DESTINATAIRE': ['APT 1', '', 'UNIT B', '', 'FLOOR 2', '', 'APT C', ''],
            'NUMERO_VOIE': ['123', '456', '789', '101', '202', '303', '404', '505'],
            'COMPLEMENT_NUMERO_VOIE': ['', '', 'B', '', '', '', 'C', ''],
            'TYPE_VOIE': ['ST', 'AVE', 'LN', 'RD', 'DR', 'ST', 'AVE', 'LN'],
            'LIBELLE_TYPE_VOIE': ['STREET', 'AVENUE', 'LANE', 'ROAD', 'DRIVE', 'STREET', 'AVENUE', 'LANE'],
            'VOIE': ['MAIN', 'OAK', 'PINE', 'ELM', 'MAPLE', 'BIRCH', 'CEDAR', 'WILLOW'],
            'LIEU_DIT': ['', 'NEAR PARK', '', 'BY THE RIVER', '', 'OLD TOWN', '', ''],
            'COMPLEMENT_ADRESSE': ['NEXT TO POST OFFICE', '', '', 'OLD MILL BUILDING', '', '', 'BEHIND STORE', ''],
            'CODE_POSTAL': ['12345', '67890', '13579', '24680', '97531', '11223', '33445', '55667'],
            'COMMUNE': ['ANYTOWN', 'SOMEWHERE', 'ELSEWHERE', 'NOWHERE', 'ANYCITY', 'TESTVILLE', 'SAMPLBURG', 'MOCKCITY']
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(DATA_FILE_PATH, sep=';', index=False)
        print(f"Dummy file {DATA_FILE_PATH} created with {len(dummy_df)} records.")

    X_raw, y_raw_df = data_loader.load_data(DATA_FILE_PATH)
    print(f"Loaded {len(X_raw)} records.")
    y_target_strings = create_target_strings(y_raw_df)
    # print("\nSample target strings:")
    # for i in range(min(3, len(y_target_strings))):
    #     print(y_target_strings[i])

    # --- 2. Tokenization and Padding ---
    print("\nTokenizing and padding data...")
    # Input Tokenizer
    input_tokenizer = Tokenizer(oov_token='<UNK>')
    input_tokenizer.fit_on_texts(X_raw)
    X_sequences = input_tokenizer.texts_to_sequences(X_raw)
    input_max_len = max(len(seq) for seq in X_sequences)
    # If all sequences are empty (e.g. very small dummy data), set a default max_len
    if input_max_len == 0 and not any(X_sequences): # handles list of empty lists
        input_max_len = 10 # Default non-zero length
    elif input_max_len == 0 and any(X_sequences): # handles list of lists like [[], [1], []]
        input_max_len = max(len(seq) if seq else 0 for seq in X_sequences)
        if input_max_len == 0: input_max_len = 10


    encoder_input_data = pad_sequences(X_sequences, maxlen=input_max_len, padding='post', truncating='post')
    input_vocab_size = len(input_tokenizer.word_index) + 1 # +1 for padding token (implicitly 0)

    # Output Tokenizer
    # filters='' to keep <START>, <END>, <SEP> tokens as part of the vocabulary
    output_tokenizer = Tokenizer(filters='', oov_token='<UNK>')
    output_tokenizer.fit_on_texts(y_target_strings)
    y_sequences = output_tokenizer.texts_to_sequences(y_target_strings)
    output_max_len = max(len(seq) for seq in y_sequences)
    if output_max_len == 0 and not any(y_sequences):
        output_max_len = 15 # Default non-zero length
    elif output_max_len == 0 and any(y_sequences):
        output_max_len = max(len(seq) if seq else 0 for seq in y_sequences)
        if output_max_len == 0: output_max_len = 15


    # Pad original y_sequences first for consistent length
    y_sequences_padded = pad_sequences(y_sequences, maxlen=output_max_len, padding='post', truncating='post')
    output_vocab_size = len(output_tokenizer.word_index) + 1

    # --- 3. Prepare Decoder Inputs/Outputs ---
    print("\nPreparing decoder inputs and outputs...")
    # Decoder input: <START> token1 token2 ... tokenN
    decoder_input_data = y_sequences_padded[:, :-1]
    # Decoder target: token1 token2 ... tokenN <END>
    decoder_target_data = y_sequences_padded[:, 1:]
    # Reshape for sparse categorical crossentropy
    decoder_target_data = np.expand_dims(decoder_target_data, -1)

    print(f"  Input vocab size: {input_vocab_size}, Max length: {input_max_len}")
    print(f"  Output vocab size: {output_vocab_size}, Max length: {output_max_len -1} (for input), {output_max_len -1} (for target)")
    print(f"  Encoder input shape: {encoder_input_data.shape}")
    print(f"  Decoder input shape: {decoder_input_data.shape}")
    print(f"  Decoder target shape: {decoder_target_data.shape}")

    if encoder_input_data.shape[0] == 0:
        print("Error: No data to train on after processing. Check CSV and tokenization.")
        return

    # --- 4. Split Data ---
    print("\nSplitting data into training and validation sets...")
    if encoder_input_data.shape[0] < 2: # train_test_split needs at least 2 samples
        print("Warning: Not enough data to create a validation set. Using all data for training.")
        encoder_input_train = encoder_input_data
        decoder_input_train = decoder_input_data
        decoder_target_train = decoder_target_data
        # Create empty validation sets to prevent training crash, though model.fit might complain
        encoder_input_val = np.array([])
        decoder_input_val = np.array([])
        decoder_target_val = np.array([])
        validation_data_tuple = None

    else:
        (encoder_input_train, encoder_input_val,
         decoder_input_train, decoder_input_val,
         decoder_target_train, decoder_target_val) = train_test_split(
            encoder_input_data, decoder_input_data, decoder_target_data,
            test_size=0.2, random_state=42, shuffle=True
        )
        validation_data_tuple = ([encoder_input_val, decoder_input_val], decoder_target_val)
        print(f"  Training samples: {encoder_input_train.shape[0]}")
        print(f"  Validation samples: {encoder_input_val.shape[0]}")


    # --- 5. Build and Train Model ---
    print("\nBuilding model...")
    # Note: output_max_len for build_model should be the length of sequences decoder *outputs*,
    # which is output_max_len - 1 because we trimmed one token for both input/target.
    training_model = model.build_model(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        input_max_len=input_max_len,
        output_max_len=output_max_len -1, # Adjusted: length of decoder_input_data/decoder_target_data sequences
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS
    )
    print("Model Summary:")
    training_model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(patience=10, monitor='val_loss', verbose=1, restore_best_weights=True) # Increased patience
    ]

    print("\nTraining model...")
    if encoder_input_train.shape[0] > 0 and (validation_data_tuple is not None and encoder_input_val.shape[0] > 0):
        history = training_model.fit(
            [encoder_input_train, decoder_input_train],
            decoder_target_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_data_tuple,
            callbacks=callbacks,
            verbose=1
        )
        print("Training complete.")
        print(f"Model saved to {MODEL_SAVE_PATH} (best performing on validation)")
    elif encoder_input_train.shape[0] > 0 : # No validation set but training data exists
        print("Training with no validation set (dataset too small).")
        history = training_model.fit(
            [encoder_input_train, decoder_input_train],
            decoder_target_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=False, monitor='loss', verbose=1)], # monitor loss
            verbose=1
        )
        print("Training complete (no validation).")
        print(f"Model saved to {MODEL_SAVE_PATH}")
    else:
        print("Skipping training as there is no training data.")


    # --- 6. Save Tokenizers ---
    print("\nSaving tokenizers...")
    with open(INPUT_TOKENIZER_PATH, 'wb') as f:
        pickle.dump(input_tokenizer, f)
    print(f"Input tokenizer saved to {INPUT_TOKENIZER_PATH}")

    with open(OUTPUT_TOKENIZER_PATH, 'wb') as f:
        pickle.dump(output_tokenizer, f)
    print(f"Output tokenizer saved to {OUTPUT_TOKENIZER_PATH}")

    # --- Cleanup dummy CSV if created ---
    if DATA_FILE_PATH == '2025_100_Adresses.csv' and "dummy_df" in locals():
        if os.path.exists(DATA_FILE_PATH):
            os.remove(DATA_FILE_PATH)
            print(f"\nCleaned up dummy data file: {DATA_FILE_PATH}")

    print("\nScript finished.")

if __name__ == '__main__':
    main()
