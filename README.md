# French Address Parser

## Description

This project implements a Conditional Random Fields (CRF) based machine learning model to parse unstructured French addresses into their structured components. It consists of two main Python scripts:

*   `train_address_parser.py`: For training the CRF model using a provided dataset.
*   `predict_address_parser.py`: For predicting structured components from new, unstructured addresses using the trained model.

## Setup

1.  **Clone the Repository (if applicable):**
    If you have downloaded this project as a ZIP, extract it. If it's a Git repository, clone it:
    ```bash
    git clone <repository_url>
    cd french-address-parser 
    ```
    *(Replace `<repository_url>` with the actual URL and `french-address-parser` with the project's directory name if different).*

2.  **Create a Virtual Environment (Recommended):**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Install the required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Data Format for Training

The training script, `train_address_parser.py`, expects a tab-separated CSV file named `address_data.csv` to be present in the root directory of the project.

The CSV file must contain the following columns:

*   `AdresseNonStructuree`: The full, unstructured address string.
*   `NuméroDansLaVoie`: The street number (e.g., "123", "71 bis").
*   `ComplementDeNumeroDeVoie`: Any complement to the street number (e.g., "B", "APT 101").
*   `LibelleTypeDeVoie`: The type of street (e.g., "RUE", "BOULEVARD", "AVENUE").
*   `NomDeVoie`: The name of the street (e.g., "DE LA PAIX", "VICTOR HUGO").
*   `LieuDit`: Locality or named place, if applicable (e.g., "LES GRANDES FERMES").
*   `CodePostal`: The postal code (e.g., "75001", "37550").
*   `Ville`: The city name (e.g., "PARIS", "ST AVERTIN").

**Important:** Even if a field is not present for a particular address, the column must exist, and the field can be left empty for that row.

**Example `address_data.csv` Snippet:**

```
AdresseNonStructuree	NuméroDansLaVoie	ComplementDeNumeroDeVoie	LibelleTypeDeVoie	NomDeVoie	LieuDit	CodePostal	Ville
71 RUE DE GRAND COUR 37550 ST AVERTIN	71		RUE	DE GRAND COUR		37550	ST AVERTIN
123B BOULEVARD DE LA LIBERTE 75001 PARIS	123	B	BOULEVARD	DE LA LIBERTE		75001	PARIS
LIEU DIT LES CHAMPS 01234 VILLAGE				LIEU DIT	LES CHAMPS	01234	VILLAGE
APT 12 45 AVENUE FOCH 69006 LYON	45	APT 12	AVENUE	FOCH		69006	LYON
```
*(Note: The example above should be tab-separated in the actual file. The training script's dummy data generator also uses this format.)*

The training script includes a feature to generate a dummy `address_data.csv` with a few examples if the file is not found, allowing for a quick demonstration. For actual training, you should provide your own comprehensive dataset in the format specified above.

## Training the Model

To train the address parsing model, run the `train_address_parser.py` script from the root directory of the project:

```bash
python train_address_parser.py
```

This script will:
1.  Attempt to load `address_data.csv`. If not found, it will create and use a small dummy dataset for demonstration.
2.  Process the data and train the CRF model.
3.  Save the trained model to a file named `address_model.crf` in the root directory.
4.  Print evaluation metrics to the console if a test set can be created (requires more than a few samples).

## Predicting Addresses

Once the model is trained and `address_model.crf` is generated, you can use `predict_address_parser.py` to parse new addresses:

```bash
python predict_address_parser.py
```

This script will:
1.  Load the `address_model.crf` file.
2.  Predict and display the structured components for a predefined list of example addresses.
3.  Enter an interactive mode where you can type an unstructured address, and the script will output the parsed components. Type `quit` to exit the interactive mode.

Example of interactive usage:
```
Loading model from address_model.crf...
Model loaded successfully.
--- Predicting Example Addresses ---
... (example predictions) ...
--- Interactive Prediction ---
Enter an address to parse (or type 'quit' to exit):
> 15 BIS RUE DES LILAS 75019 PARIS
Parsed:
  NuméroDansLaVoie: 15 BIS
  LibelleTypeDeVoie: RUE
  NomDeVoie: DES LILAS
  CodePostal: 75019
  Ville: PARIS
--------------------
> quit
Script execution finished.
```

## Dependencies

All Python dependencies required for this project are listed in the `requirements.txt` file. They include:

*   `pandas`
*   `scikit-learn`
*   `sklearn-crfsuite`
*   `joblib`

These can be installed as described in the Setup section.
