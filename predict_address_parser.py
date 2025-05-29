import joblib
import re

# Copied from train_address_parser.py
FIELD_TAGS = {
    'NuméroDansLaVoie': 'NumeroDansLaVoie',
    'ComplementDeNumeroDeVoie': 'ComplementDeNumeroDeVoie',
    'LibelleTypeDeVoie': 'LibelleTypeDeVoie',
    'NomDeVoie': 'NomDeVoie',
    'LieuDit': 'LieuDit',
    'CodePostal': 'CodePostal',
    'Ville': 'Ville'
}

# Copied from train_address_parser.py
def tokenize(text):
    """
    Tokenizes text into words using a simple regex.
    Handles punctuation and hyphens.
    """
    if not isinstance(text, str):
        return []
    return re.findall(r"[\w'-]+|[.,;!?()]", str(text))

# Copied from train_address_parser.py
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
        'postag': '', 
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
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
        })
    else:
        features['EOS'] = True
    
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

# Copied from train_address_parser.py
def sent2features(sent):
    """Applies word2features to each token in a sentence."""
    return [word2features(sent, i) for i in range(len(sent))]

def load_model(model_path):
    """
    Loads the CRF model from the specified path.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def parse_predicted_tags(tokens, tags):
    """
    Reconstructs structured address components from tokens and BIO tags.
    """
    structured_address = {}
    current_field_tokens = []
    current_field_type = None

    # Invert FIELD_TAGS for easier lookup from tag to field name
    tag_to_field_name = {v: k for k, v in FIELD_TAGS.items()}

    for token, tag in zip(tokens, tags):
        if tag == 'O':
            if current_field_type: # End of a field
                field_name = tag_to_field_name.get(current_field_type)
                if field_name:
                    value = " ".join(current_field_tokens)
                    structured_address[field_name] = structured_address.get(field_name, "") + value + " "
                current_field_tokens = []
                current_field_type = None
            continue

        tag_prefix, tag_type = tag.split('-', 1)

        if tag_prefix == 'B':
            if current_field_type: # End of previous field
                field_name = tag_to_field_name.get(current_field_type)
                if field_name:
                    value = " ".join(current_field_tokens)
                    structured_address[field_name] = structured_address.get(field_name, "") + value + " "
            
            current_field_tokens = [token]
            current_field_type = tag_type
        
        elif tag_prefix == 'I':
            if current_field_type == tag_type: # Continuation of the current field
                current_field_tokens.append(token)
            else: # Unexpected I-tag without a B-tag or matching B-tag
                if current_field_type: # End of previous field
                    field_name = tag_to_field_name.get(current_field_type)
                    if field_name:
                        value = " ".join(current_field_tokens)
                        structured_address[field_name] = structured_address.get(field_name, "") + value + " "
                # Treat this I-tag as a new B-tag for robustness
                current_field_tokens = [token]
                current_field_type = tag_type
    
    # After loop, add any remaining field
    if current_field_type and current_field_tokens:
        field_name = tag_to_field_name.get(current_field_type)
        if field_name:
            value = " ".join(current_field_tokens)
            structured_address[field_name] = structured_address.get(field_name, "") + value + " "

    # Trim trailing spaces from values
    for key in structured_address:
        structured_address[key] = structured_address[key].strip()
        
    return structured_address


def predict_address(model, unstructured_address_text):
    """
    Predicts structured address components from an unstructured address string.
    """
    if not unstructured_address_text or not isinstance(unstructured_address_text, str):
        return {}
        
    tokens = tokenize(unstructured_address_text)
    if not tokens:
        return {}

    features = sent2features(tokens)
    predicted_tags = model.predict_single(features)
    
    # Debug: Print tokens and predicted tags
    # print(f"Debug Tokens: {tokens}")
    # print(f"Debug Tags:   {predicted_tags}")

    structured_result = parse_predicted_tags(tokens, predicted_tags)
    return structured_result

if __name__ == '__main__':
    MODEL_FILE_PATH = 'address_model.crf'
    
    print(f"Loading model from {MODEL_FILE_PATH}...")
    crf_model = load_model(MODEL_FILE_PATH)

    if crf_model:
        print("Model loaded successfully.")

        example_addresses = [
            "123 RUE DE LA PAIX 75002 PARIS",
            "APT C21 5 BOULEVARD VICTOR HUGO 06000 NICE",
            "LIEU DIT LES GRANDES FERMES 37250 VEIGNE",
            "71 RUE DE GRAND COUR 37550 ST AVERTIN",
            "CS 50001 5 RUE DE LA PAIX 37000 TOURS CEDEX 1",
            "1 BIS AVENUE DES CHAMPS ELYSEES 75008 PARIS",
            "MAIRIE DE PLOUZANE 29280 PLOUZANE" # Example without explicit street type
        ]

        print("\n--- Predicting Example Addresses ---")
        for address in example_addresses:
            print(f"\nOriginal: \"{address}\"")
            parsed_address = predict_address(crf_model, address)
            if parsed_address:
                print("Parsed:")
                for field, value in parsed_address.items():
                    print(f"  {field}: {value}")
            else:
                print("  Could not parse address (no tokens or empty input).")
        
        print("\n--- Interactive Prediction ---")
        print("Enter an address to parse (or type 'quit' to exit):")
        while True:
            user_input = input("> ")
            if user_input.lower() == 'quit':
                break
            if not user_input.strip():
                continue
            
            parsed_address = predict_address(crf_model, user_input)
            if parsed_address:
                print("Parsed:")
                for field, value in parsed_address.items():
                    print(f"  {field}: {value}")
            else:
                print("  Could not parse address (no tokens or empty input).")
            print("-" * 20)
            
    else:
        print(f"Could not load model. Ensure '{MODEL_FILE_PATH}' exists and was created by the training script.")
        print("You may need to run the 'train_address_parser.py' script first to generate the model file.")

print("\nScript execution finished.")
