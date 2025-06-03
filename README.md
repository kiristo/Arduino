# French Address Parser

## Description

This project implements a sequence-to-sequence (Seq2Seq) machine learning model to parse unstructured French addresses into their structured components. The model takes a single French address string as input and outputs a predefined set of address fields. This project is built using Python, TensorFlow/Keras, Pandas, and Scikit-learn.

## Input Data

The model is trained on CSV files containing unstructured addresses and their corresponding structured components.

The CSV file should have the following structure:
*   **Input Column**: `AdresseNonStructure` (the full, unstructured address string).
*   **Output Columns**: These are the target fields the model learns to predict.
    *   `COMPLEMENT_DESTINATAIRE`
    *   `NUMERO_VOIE`
    *   `COMPLEMENT_NUMERO_VOIE`
    *   `TYPE_VOIE`
    *   `LIBELLE_TYPE_VOIE`
    *   `VOIE`
    *   `LIEU_DIT`
    *   `COMPLEMENT_ADRESSE`
    *   `CODE_POSTAL`
    *   `COMMUNE`

Sample data files provided:
*   `2025_100_Adresses.csv`: A small dataset with 100 addresses, primarily for quick testing and demonstration. The `train.py` script creates a dummy version of this with 8 records if it's not found.
*   `2025_100k_Adresses.csv`: A larger dataset with 100,000 addresses for more robust model training. (Note: This file is not included in the repository by default due to its size but is expected to be present for full training.)

## Setup

1.  **Create a Virtual Environment (Recommended)**:
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

To train the address parsing model:

1.  **Run the Training Script**:
    ```bash
    python train.py
    ```
2.  **Dataset**:
    *   By default, `train.py` is configured to use the `2025_100_Adresses.csv` file. If this file is not present, the script will automatically generate a small dummy version of it with 8 records for testing purposes.
    *   To train on the larger dataset, modify the `DATA_FILE_PATH` variable in `train.py` to point to `2025_100k_Adresses.csv`:
        ```python
        # In train.py
        DATA_FILE_PATH = '2025_100k_Adresses.csv'
        ```
3.  **Output**:
    The training process will save the following files:
    *   `address_parser_model.keras`: The trained Keras model.
    *   `input_tokenizer.pkl`: The tokenizer for pre-processing input addresses.
    *   `output_tokenizer.pkl`: The tokenizer for processing output structured fields.

## Making Predictions

To use the trained model to parse new addresses:

1.  **Run the Prediction Script**:
    Ensure that the model (`address_parser_model.keras`) and tokenizers (`input_tokenizer.pkl`, `output_tokenizer.pkl`) from the training step are present in the project directory.
    ```bash
    python predict.py
    ```
2.  **Functionality**:
    *   The `predict.py` script loads the saved model and tokenizers.
    *   It contains a list of sample unstructured addresses.
    *   For each sample address, it preprocesses the input, predicts the structured components, and prints the formatted output.
3.  **Output Fields**:
    The predicted structured components correspond to the following fields:
    *   `COMPLEMENT_DESTINATAIRE`
    *   `NUMERO_VOIE`
    *   `COMPLEMENT_NUMERO_VOIE`
    *   `TYPE_VOIE`
    *   `LIBELLE_TYPE_VOIE`
    *   `VOIE`
    *   `LIEU_DIT`
    *   `COMPLEMENT_ADRESSE`
    *   `CODE_POSTAL`
    *   `COMMUNE`

    *Note: The quality of predictions depends heavily on the size and quality of the training data, and the training duration. Predictions from a model trained on the small dummy dataset will not be meaningful.*

## Scripts Overview

*   **`data_loader.py`**:
    *   Handles loading data from the input CSV file.
    *   Performs initial preprocessing, such as handling encoding issues and preparing input (X) and output (y) data structures.
*   **`model.py`**:
    *   Defines the sequence-to-sequence (Seq2Seq) neural network architecture using TensorFlow and Keras.
    *   Includes functions to build, save, and load the Keras model.
*   **`train.py`**:
    *   Orchestrates the entire model training pipeline.
    *   Loads data using `data_loader.py`.
    *   Prepares and tokenizes input and output sequences, including special tokens (`<START>`, `<END>`, `<SEP>`).
    *   Splits data into training and validation sets.
    *   Builds the model using `model.py`.
    *   Trains the model and saves the trained model and tokenizers.
*   **`predict.py`**:
    *   Loads the trained model and tokenizers.
    *   Reconstructs the encoder and decoder parts of the Seq2Seq model for inference.
    *   Provides functions to predict structured components from new unstructured address strings.
    *   Includes example usage with sample addresses.
*   **`requirements.txt`**:
    *   Lists the Python dependencies required for the project (Pandas, TensorFlow, Scikit-learn).

This project provides a foundational framework for parsing French addresses. Further improvements could include more sophisticated preprocessing, hyperparameter tuning, and evaluation metrics.
