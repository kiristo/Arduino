import pandas as pd

def load_data(file_path: str):
    """
    Loads data from a CSV file, preprocesses it, and separates it into input (X) and output (y) columns.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing two pandas Series/DataFrames:
               X (unstructured addresses) and y (structured addresses).
    """
    try:
        df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, delimiter=';', encoding='latin1')

    input_col = 'AdresseNonStructure'
    output_cols = [
        'COMPLEMENT_DESTINATAIRE', 'NUMERO_VOIE', 'COMPLEMENT_NUMERO_VOIE',
        'TYPE_VOIE', 'LIBELLE_TYPE_VOIE', 'VOIE', 'LIEU_DIT',
        'COMPLEMENT_ADRESSE', 'CODE_POSTAL', 'COMMUNE'
    ]

    # Preprocess output columns: fill NaN values with an empty string
    for col in output_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            # If an output column doesn't exist, create it and fill with empty strings
            df[col] = ''


    # Preprocess input column: fill NaN values with an empty string and strip whitespace
    if input_col in df.columns:
        df[input_col] = df[input_col].fillna('').str.strip()
    else:
        # If the input column doesn't exist, raise an error or handle as appropriate
        # For now, let's assume it must exist and raise an error if not.
        raise ValueError(f"Input column '{input_col}' not found in the CSV file.")

    # Strip whitespace from output columns
    for col in output_cols:
        if col in df.columns and df[col].dtype == 'object': # Apply only to text-like columns
            df[col] = df[col].str.strip()

    X = df[input_col]
    y = df[output_cols]

    return X, y

if __name__ == '__main__':
    # Assuming '2025_100_Adresses.csv' is in the same directory as the script
    # or provide the full/relative path to the CSV file.
    # For the purpose of this example, let's create a dummy CSV file
    # In a real scenario, this file would already exist.
    dummy_data = {
        'AdresseNonStructure': [
            '  123 MAIN ST, ANYTOWN  ',
            '456 OAK AVE, SOMEWHERE',
            '  789 PINE LN, ELSEWHERE  ',
            '101 ELM RD, NOWHERE',
            '202 MAPLE DR, ANYCITY'
        ],
        'COMPLEMENT_DESTINATAIRE': ['APT 1', '', 'UNIT B', '', 'FLOOR 2'],
        'NUMERO_VOIE': ['123', '456', '789', '101', '202'],
        'COMPLEMENT_NUMERO_VOIE': ['', '', 'B', '', ''],
        'TYPE_VOIE': ['ST', 'AVE', 'LN', 'RD', 'DR'],
        'LIBELLE_TYPE_VOIE': ['STREET', 'AVENUE', 'LANE', 'ROAD', 'DRIVE'],
        'VOIE': ['MAIN', 'OAK', 'PINE', 'ELM', 'MAPLE'],
        'LIEU_DIT': ['', 'NEAR PARK', '', 'BY THE RIVER', ''],
        'COMPLEMENT_ADRESSE': ['NEXT TO POST OFFICE', '', '', 'OLD MILL BUILDING', ''],
        'CODE_POSTAL': ['12345', '67890', '13579', '24680', '97531'],
        'COMMUNE': ['ANYTOWN', 'SOMEWHERE', 'ELSEWHERE', 'NOWHERE', 'ANYCITY'],
        'EXTRA_COLUMN_TO_IGNORE': ['A', 'B', 'C', 'D', 'E']
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = '2025_100_Adresses.csv'
    dummy_df.to_csv(dummy_csv_path, sep=';', index=False)

    print(f"Loading data from: {dummy_csv_path}")
    try:
        X_loaded, y_loaded = load_data(dummy_csv_path)
        print("\nFirst 5 X records (Unstructured Addresses):")
        print(X_loaded.head())
        print("\nFirst 5 y records (Structured Address Components):")
        print(y_loaded.head())

        # Verify NaN filling and stripping for a specific case if possible
        print("\nVerifying preprocessing for a potentially problematic original entry:")
        # Create a more specific dummy CSV for detailed verification
        problematic_data = {
            'AdresseNonStructure': ['  Address with spaces  ', pd.NA],
            'COMPLEMENT_DESTINATAIRE': [pd.NA, '  Comp with spaces  '],
            'NUMERO_VOIE': ['123', '456'],
            'COMPLEMENT_NUMERO_VOIE': [pd.NA, pd.NA],
            'TYPE_VOIE': ['ST', 'AVE'],
            'LIBELLE_TYPE_VOIE': ['STREET', 'AVENUE'],
            'VOIE': ['MAIN', 'OAK'],
            'LIEU_DIT': [pd.NA, ''],
            'COMPLEMENT_ADRESSE': ['', pd.NA],
            'CODE_POSTAL': ['12345', '67890'],
            'COMMUNE': ['ANYTOWN', 'SOMEWHERE']
        }
        problematic_df = pd.DataFrame(problematic_data)
        problematic_csv_path = 'problematic_test_data.csv'
        problematic_df.to_csv(problematic_csv_path, sep=';', index=False)

        X_prob, y_prob = load_data(problematic_csv_path)
        print("\nX from problematic data:")
        print(X_prob)
        print("\ny from problematic data:")
        print(y_prob)
        # Check specific conditions
        assert X_prob.iloc[0] == 'Address with spaces' # Stripped
        assert X_prob.iloc[1] == '' # NaN to empty string
        assert y_prob['COMPLEMENT_DESTINATAIRE'].iloc[0] == '' # NaN to empty string
        assert y_prob['COMPLEMENT_DESTINATAIRE'].iloc[1] == 'Comp with spaces' # Stripped
        print("\nPreprocessing assertions passed for problematic data.")

    except FileNotFoundError:
        print(f"Error: The file '{dummy_csv_path}' was not found. Please ensure it's in the correct directory.")
    except ValueError as ve:
        print(f"ValueError during data loading: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Clean up dummy files
    import os
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)
    if os.path.exists(problematic_csv_path):
        os.remove(problematic_csv_path)
    print("\nCleaned up dummy CSV files.")
