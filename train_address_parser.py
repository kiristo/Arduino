import pandas as pd
from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import train_test_split
import joblib
import re

# Define the expected columns for the CSV file
EXPECTED_COLUMNS = [
    'AdresseNonStructuree', 'NuméroDansLaVoie', 'ComplementDeNumeroDeVoie',
    'LibelleTypeDeVoie', 'NomDeVoie', 'LieuDit', 'CodePostal', 'Ville'
]

# Define BIO tags for structured fields
FIELD_TAGS = {
    'NuméroDansLaVoie': 'NumeroDansLaVoie',
    'ComplementDeNumeroDeVoie': 'ComplementDeNumeroDeVoie',
    'LibelleTypeDeVoie': 'LibelleTypeDeVoie',
    'NomDeVoie': 'NomDeVoie',
    'LieuDit': 'LieuDit',
    'CodePostal': 'CodePostal',
    'Ville': 'Ville'
}

def load_data(file_path):
    """
    Reads a tab-separated CSV into a pandas DataFrame.
    Expected columns: AdresseNonStructuree, NuméroDansLaVoie, ComplementDeNumeroDeVoie,
                      LibelleTypeDeVoie, NomDeVoie, LieuDit, CodePostal, Ville.
    Handles potential FileNotFoundError.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
        # Validate columns
        if not all(col in df.columns for col in EXPECTED_COLUMNS):
            raise ValueError(f"CSV file must contain the following columns: {EXPECTED_COLUMNS}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def tokenize(text):
    """
    Tokenizes text into words using a simple regex.
    Handles punctuation and hyphens.
    """
    if not isinstance(text, str):
        return []
    # return re.findall(r"[\w'-]+|\S", str(text)) # \S captures punctuation as separate tokens
    return re.findall(r"[\w'-]+|[.,;!?()]", str(text)) # Keep common punctuation as separate tokens

def create_training_data(df):
    """
    Takes the DataFrame and returns two lists: X (list of lists of tokens)
    and y (list of lists of BIO tags).
    """
    X = []
    y = []

    for _, row in df.iterrows():
        unstructured_address = str(row['AdresseNonStructuree'])
        unstructured_tokens = tokenize(unstructured_address)
        
        if not unstructured_tokens:
            continue

        tags = ['O'] * len(unstructured_tokens)
        current_token_idx = 0

        # Order of fields matters for greedy matching
        ordered_fields = [
            'NuméroDansLaVoie', 'ComplementDeNumeroDeVoie', 'LibelleTypeDeVoie', 
            'NomDeVoie', 'LieuDit', 'CodePostal', 'Ville'
        ]

        # Store structured field tokens for matching
        structured_field_tokens = {}
        for field in ordered_fields:
            field_value = str(row[field])
            if field_value: # Ensure there's a value
                structured_field_tokens[field] = tokenize(field_value)

        # Greedy matching approach
        # Iterate through unstructured_tokens and try to match sequences with structured fields
        # This is a simplified initial approach. More robust matching might be needed.
        
        temp_tags = ['O'] * len(unstructured_tokens)
        
        # Create a dictionary to hold the tokenized versions of the structured fields
        structured_data = {}
        for col_name, bio_base_tag in FIELD_TAGS.items():
            if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip():
                structured_data[bio_base_tag] = tokenize(str(row[col_name]))
        
        # Attempt to tag based on sequence matching
        # This is a challenging part and might need several iterations to get right.
        # A common approach is to iterate through the unstructured tokens and try to match
        # the longest possible sequence from the structured fields.

        # Simplified greedy matching from left to right
        # This initial version will try to find the first occurrence of a field's tokens
        # in the unstructured address. More advanced logic would be needed for overlapping
        # or out-of-order elements.

        unstructured_idx = 0
        while unstructured_idx < len(unstructured_tokens):
            matched_field = False
            for field_name in ordered_fields: # Iterate in predefined order
                field_tokens = structured_field_tokens.get(field_name)
                if not field_tokens:
                    continue

                # Check if the current sequence of unstructured tokens matches the field_tokens
                if unstructured_idx + len(field_tokens) <= len(unstructured_tokens) and \
                   all(unstructured_tokens[unstructured_idx + k].lower() == field_tokens[k].lower() for k in range(len(field_tokens))):
                    
                    base_tag = FIELD_TAGS[field_name]
                    tags[unstructured_idx] = f'B-{base_tag}'
                    for k in range(1, len(field_tokens)):
                        tags[unstructured_idx + k] = f'I-{base_tag}'
                    
                    unstructured_idx += len(field_tokens)
                    matched_field = True
                    # Remove matched field to prevent re-matching (simplistic)
                    structured_field_tokens[field_name] = [] 
                    break # Move to next unstructured token after a match
            
            if not matched_field:
                unstructured_idx += 1 # Move to the next token if no match found

        X.append(unstructured_tokens)
        y.append(tags)
        
    return X, y

def word2features(sent, i):
    """
    Extracts features for a token in a sentence.
    sent: list of tokens
    i: index of the token
    """
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        # 'word.shape': word_shape(word), # Placeholder for word shape feature
        'postag': '', # Placeholder for POS tag if available
        'word_len': len(word),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
        })
    else:
        features['BOS'] = True # Beginning of Sentence

    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
        })
    else:
        features['EOS'] = True # End of Sentence
    
    # Features for words at distance 2
    if i > 1:
        word2 = sent[i-2]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.isdigit()': word2.isdigit(),
        })
    
    if i < len(sent)-2:
        word2 = sent[i+2]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.isdigit()': word2.isdigit(),
        })

    return features

def sent2features(sent):
    """Applies word2features to each token in a sentence."""
    return [word2features(sent, i) for i in range(len(sent))]

if __name__ == '__main__':
    DATA_FILE_PATH = "address_data.csv" # Will be created in a later step

    # Create a dummy address_data.csv for testing
    dummy_data = {
        'AdresseNonStructuree': [
            "71 RUE DE GRAND COUR 37550 ST AVERTIN",
            "123B BOULEVARD DE LA LIBERTE 75001 PARIS",
            "LIEU DIT LES CHAMPS 01234 VILLAGE",
            "APT 12 45 AVENUE FOCH 69006 LYON",
            "CS 50001 5 RUE DE LA PAIX 37000 TOURS CEDEX 1" 
        ],
        'NuméroDansLaVoie': ["71", "123", "", "45", "5"],
        'ComplementDeNumeroDeVoie': ["", "B", "", "APT 12", ""], # Note: 'APT 12' is more complex
        'LibelleTypeDeVoie': ["RUE", "BOULEVARD", "", "AVENUE", "RUE"],
        'NomDeVoie': ["DE GRAND COUR", "DE LA LIBERTE", "", "FOCH", "DE LA PAIX"],
        'LieuDit': ["", "", "LES CHAMPS", "", ""],
        'CodePostal': ["37550", "75001", "01234", "69006", "37000"],
        'Ville': ["ST AVERTIN", "PARIS", "VILLAGE", "LYON", "TOURS CEDEX 1"]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(DATA_FILE_PATH, sep='\t', index=False)
    print(f"Dummy '{DATA_FILE_PATH}' created for demonstration.")

    df = load_data(DATA_FILE_PATH)

    if df is not None:
        print("Data loaded successfully.")
        X_tokens, y_tags = create_training_data(df)
        
        # Output some examples of tokenization and tagging for verification
        print("\nSample tokenization and tagging:")
        for i in range(min(3, len(X_tokens))): # Print first 3 samples
            print(f"Address: {df['AdresseNonStructuree'].iloc[i]}")
            print(f"Tokens: {X_tokens[i]}")
            print(f"Tags:   {y_tags[i]}")
            print("-" * 30)

        if not X_tokens or not any(X_tokens): # Check if X_tokens is empty or contains empty lists
            print("No tokens were generated. Please check the input data and tokenization logic.")
        else:
            X_features = [sent2features(s) for s in X_tokens if s] # Ensure sentence is not empty

            if not X_features:
                print("No features were generated. Please check sent2features and tokenization.")
            else:
                # Ensure y_tags corresponds to non-empty X_tokens
                y_tags_filtered = [y_tags[i] for i, s in enumerate(X_tokens) if s]

                if len(X_features) != len(y_tags_filtered):
                    print(f"Mismatch between features ({len(X_features)}) and tags ({len(y_tags_filtered)}) after filtering. Aborting.")
                elif not y_tags_filtered:
                     print("No tags available for training after filtering. Aborting.")
                else:
                    print(f"\nNumber of samples for training: {len(X_features)}")

                    # Split data
                    # Ensure there's enough data to split
                    if len(X_features) < 2:
                        print("Not enough data to split into training and testing sets. Skipping model training.")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_features, y_tags_filtered, test_size=0.2, random_state=42
                        )

                        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

                        # Initialize CRF model
                        crf = CRF(
                            algorithm='lbfgs',
                            c1=0.1,
                            c2=0.1,
                            max_iterations=100,
                            all_possible_transitions=True
                        )

                        print("\nTraining CRF model...")
                        try:
                            crf.fit(X_train, y_train)
                            print("Model training completed.")

                            # Make predictions
                            y_pred = crf.predict(X_test)

                            # Evaluate (excluding 'O' tags from report for clarity)
                            labels = sorted(
                                list(set(tag for sent_tags in y_test for tag in sent_tags if tag != 'O'))
                            )
                            if not labels:
                                print("\nNo non-'O' tags found in the test set for evaluation.")
                            else:
                                print("\nClassification Report (excluding 'O' tags):")
                                # Ensure y_test and y_pred are not empty and have the same structure
                                if y_test and y_pred and len(y_test) == len(y_pred):
                                     print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))
                                else:
                                    print("Could not generate classification report due to empty or mismatched y_test/y_pred.")


                            # Save the model
                            MODEL_FILE_PATH = 'address_model.crf'
                            joblib.dump(crf, MODEL_FILE_PATH)
                            print(f"\nTrained model saved to {MODEL_FILE_PATH}")

                        except Exception as e:
                            print(f"An error occurred during model training or evaluation: {e}")
                            # Print some details about the data for debugging
                            print(f"Number of training features: {len(X_train)}")
                            if X_train: print(f"Example training features item: {X_train[0][0] if X_train[0] else 'Empty sentence'}")
                            print(f"Number of training tags: {len(y_train)}")
                            if y_train: print(f"Example training tags item: {y_train[0]}")


    else:
        print(f"Could not load data from {DATA_FILE_PATH}. Script cannot proceed.")

print("\nScript execution finished.")
