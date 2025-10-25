import pandas as pd
import re
import string

def extract_value(text):
    match = re.search(r'Value:\s*([\d.]+)', text)
    return float(match.group(1)) if match else None

def extract_unit(text):
    match = re.search(r'Unit:\s*([A-Za-z ]+)', text)
    return match.group(1).strip() if match else None

def count_sentences(text):
    return len(re.findall(r'[.!?]', text))

def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

def has_special_chars(text):
    return int(bool(re.search(r'[^A-Za-z0-9\s]', text)))

def uppercase_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0
    upper = [c for c in letters if c.isupper()]
    return len(upper) / len(letters)

def create_features(df):
    df['is_organic'] = df['catalog_content'].str.contains(r'\borganic\b', flags=re.IGNORECASE, regex=True).astype(int)
    df['is_gourmet'] = df['catalog_content'].str.contains(r'\b(gourmet|premium|luxury)\b', flags=re.IGNORECASE, regex=True).astype(int)
    df['is_gluten_free'] = df['catalog_content'].str.contains(r'\bgluten[-\s]?free\b', flags=re.IGNORECASE, regex=True).astype(int)
    df['unit_type'] = df['catalog_content'].apply(extract_unit)
    df['value'] = df['catalog_content'].apply(extract_value)
    df['num_sentences'] = df['catalog_content'].apply(count_sentences)
    df['num_words'] = df['catalog_content'].apply(count_words)
    df['has_special_chars'] = df['catalog_content'].apply(has_special_chars)
    df['uppercase_ratio'] = df['catalog_content'].apply(uppercase_ratio)
    df['is_bulk'] = df['catalog_content'].str.contains(r'\b(bulk|pack of|pack)\b', flags=re.IGNORECASE, regex=True).astype(int)
    df['is_bulk'] = ((df['is_bulk'] == 1) | (df['value'] > 50)).astype(int)
    return df

def preprocess(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_features = create_features(train.copy())
    test_features = create_features(test.copy())

    keep_cols_train = ['sample_id', 'catalog_content', 'price', 'is_organic', 'is_gourmet', 
                       'is_gluten_free', 'unit_type', 'value', 'num_sentences', 
                       'num_words', 'has_special_chars', 'uppercase_ratio', 'is_bulk']

    keep_cols_test = ['sample_id', 'catalog_content', 'is_organic', 'is_gourmet', 
                      'is_gluten_free', 'unit_type', 'value', 'num_sentences', 
                      'num_words', 'has_special_chars', 'uppercase_ratio', 'is_bulk']

    train_features = train_features[keep_cols_train]
    test_features = test_features[keep_cols_test]

    train_features.to_csv('train_processed.csv', index=False)
    test_features.to_csv('test_processed.csv', index=False)

    print("✅ train_processed.csv and test_processed.csv created successfully.")
    return train_features, test_features


train_df, test_df = preprocess("train.csv", "test.csv")
print("Ending Preprocessing Step.")
