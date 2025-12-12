import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import csv

def load_data(file_path):
    """Loads data from a TSV file."""
    # Assuming tab-separated, no header, and specific columns based on preprocess.py
    # Columns are 'score', 'joke', 'body', 'punchline'
    df = pd.read_csv(file_path, sep='\t', header=None, names=['score', 'joke', 'body', 'punchline'], quoting=csv.QUOTE_NONE, escapechar='\\')
    # Ensure text columns are strings and fill NaNs
    for col in ['joke', 'body', 'punchline']:
        df[col] = df[col].fillna("").astype(str)
    return df

def generate_embeddings(model, sentences):
    """Generates SBERT embeddings for a list of sentences."""
    return model.encode(sentences, show_progress_bar=True)

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    # train_file = os.path.join(data_dir, "train.tsv")                                                                                                                               
    # dev_file = os.path.join(data_dir, "dev.tsv")                                                                                                                                  
    # test_file = os.path.join(data_dir, "test.tsv")
    train_file = os.path.join(data_dir, "train_2015.tsv")
    dev_file = os.path.join(data_dir, "test_2019.tsv")
    test_file = os.path.join(data_dir, "test_2019.tsv")


    train_df = load_data(train_file)
    dev_df = load_data(dev_file)
    test_df = load_data(test_file)

    print("Data loaded:")
    print(f"Train samples: {len(train_df)}")
    print(f"Dev samples: {len(dev_df)}")
    print(f"Test samples: {len(test_df)}")

    sbert_model_name = 'all-mpnet-base-v2'
    print(f"\nLoading SBERT model: {sbert_model_name}")
    sbert_model = SentenceTransformer(sbert_model_name)

    # --- Method 1: Single embedding for the whole joke ---
    print("\n--- Generating embeddings (Method 1: Single embedding) ---")
    X_train_m1 = generate_embeddings(sbert_model, train_df['joke'].tolist())
    X_dev_m1 = generate_embeddings(sbert_model, dev_df['joke'].tolist())
    X_test_m1 = generate_embeddings(sbert_model, test_df['joke'].tolist())
    y_train = train_df['score'].values
    y_dev = dev_df['score'].values
    y_test = test_df['score'].values

    # Scale the features
    scaler_m1 = StandardScaler()
    X_train_m1_scaled = scaler_m1.fit_transform(X_train_m1)
    X_dev_m1_scaled = scaler_m1.transform(X_dev_m1)
    X_test_m1_scaled = scaler_m1.transform(X_test_m1)

    print("\n--- Training and evaluating models (Method 1) ---")
    models_m1 = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
    }

    for name, model in models_m1.items():
        print(f"\nTraining {name} for Method 1...")
        model.fit(X_train_m1_scaled, y_train)
        y_pred = model.predict(X_test_m1_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        spearman, _ = spearmanr(y_test, y_pred)

        print(f"Results for {name} (Method 1):")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R^2: {r2:.4f}")
        print(f"  Spearman's Rho: {spearman:.4f}")

    # --- Method 2: Separate setup and punchline embeddings ---
    print("\n--- Generating embeddings (Method 2: Separate setup/punchline) ---")
    setup_train_emb = generate_embeddings(sbert_model, train_df['body'].tolist())
    punchline_train_emb = generate_embeddings(sbert_model, train_df['punchline'].tolist())
    setup_dev_emb = generate_embeddings(sbert_model, dev_df['body'].tolist())
    punchline_dev_emb = generate_embeddings(sbert_model, dev_df['punchline'].tolist())
    setup_test_emb = generate_embeddings(sbert_model, test_df['body'].tolist())
    punchline_test_emb = generate_embeddings(sbert_model, test_df['punchline'].tolist())

    # calculate cosine similarity/distance (1 - cosine_similarity for distance)
    # cosine similarity: (A . B) / (||A|| * ||B||)
    def cosine_similarity(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def calculate_cosine_distances(setup_embeddings, punchline_embeddings):
        distances = []
        for i in range(len(setup_embeddings)):
            dist = 1 - cosine_similarity(setup_embeddings[i], punchline_embeddings[i])
            distances.append(dist)
        return np.array(distances).reshape(-1, 1)

    cosine_distances_train = calculate_cosine_distances(setup_train_emb, punchline_train_emb)
    cosine_distances_dev = calculate_cosine_distances(setup_dev_emb, punchline_dev_emb)
    cosine_distances_test = calculate_cosine_distances(setup_test_emb, punchline_test_emb)

    # Concatenate setup embeddings, punchline embeddings, and cosine distance
    X_train_m2 = np.concatenate((setup_train_emb, punchline_train_emb, cosine_distances_train), axis=1)
    X_dev_m2 = np.concatenate((setup_dev_emb, punchline_dev_emb, cosine_distances_dev), axis=1)
    X_test_m2 = np.concatenate((setup_test_emb, punchline_test_emb, cosine_distances_test), axis=1)

    # Scale the features
    scaler_m2 = StandardScaler()
    X_train_m2_scaled = scaler_m2.fit_transform(X_train_m2)
    X_dev_m2_scaled = scaler_m2.transform(X_dev_m2)
    X_test_m2_scaled = scaler_m2.transform(X_test_m2)

    print("\n--- Training and evaluating models (Method 2) ---")
    models_m2 = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
    }

    for name, model in models_m2.items():
        print(f"\nTraining {name} for Method 2...")
        model.fit(X_train_m2_scaled, y_train)
        y_pred = model.predict(X_test_m2_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        spearman, _ = spearmanr(y_test, y_pred)

        print(f"Results for {name} (Method 2):")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R^2: {r2:.4f}")
        print(f"  Spearman's Rho: {spearman:.4f}")


if __name__ == "__main__":
    main()
