import json
import os
import pandas as pd
import numpy as np
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def phrase_in_doc(phrase, updated_doc):
    """ A helper function to see if the phrase is in the document body (title) or punchline (selftext) """
    return phrase in updated_doc["title"] or phrase in updated_doc["selftext"]

def create_2015_2019_splits(file_name: str = "fullrjokes.json"):
    print("Reading and preprocessing for 2015/2019 split...")
    
    # Time ranges
    START_2015 = 1420070400 # 2015-01-01
    END_2015 = 1451606400   # 2016-01-01
    
    START_2019 = 1546300800 # 2019-01-01
    END_2019 = 1577836800   # 2020-01-01

    list_of_jokes = []
    
    with open(os.path.join(PROJECT_ROOT, "data", file_name), "r") as f:
        for line in f:
            updated_doc = json.loads(line)
            created = updated_doc["created_utc"]
            
            # Filter by year first to save processing
            is_2015 = START_2015 <= created < END_2015
            is_2019 = START_2019 <= created < END_2019
            
            if not (is_2015 or is_2019):
                continue

            # Standard filters
            if updated_doc["title"] == "" and updated_doc["selftext"] == "":
                continue
            if phrase_in_doc("[deleted]", updated_doc):
                continue
            if phrase_in_doc("[removed]", updated_doc):
                continue
            if pd.isnull(updated_doc["score"]) or (type(updated_doc["score"]) == str and updated_doc["score"] == ""):
                continue
            if phrase_in_doc("[pic]", updated_doc) or phrase_in_doc("VIDEO", updated_doc):
                continue

            joke = updated_doc["title"] + " " + updated_doc["selftext"]
            list_of_jokes.append({
                "joke": joke, 
                "body": updated_doc["title"], 
                "punchline": updated_doc["selftext"], 
                "score": updated_doc["score"], 
                "date": created,
                "year": 2015 if is_2015 else 2019
            })
                        
    df = pd.DataFrame(list_of_jokes)

    # Cleanup text
    for col in ["joke", "body", "punchline"]:
        df[col] = df[col].astype(str).replace(r'\n',' ',regex=True)
        df[col] = df[col].replace(r'\r',' ',regex=True)
        df[col] = df[col].replace(r'\t',' ',regex=True)

    df = df.dropna() # This line was missing a newline
    
    # Log score transformation
    df["score"] = np.log1p(df["score"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df["score"] = df["score"].fillna(0)
    
    # Split
    train_df = df[df["year"] == 2015]
    test_df = df[df["year"] == 2019]
    
    print(f"Found {len(train_df)} jokes for 2015 (Train)")
    print(f"Found {len(test_df)} jokes for 2019 (Test)")
    
    # Save
    train_path = os.path.join(PROJECT_ROOT, "data", "train_2015.tsv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test_2019.tsv")
    
    # Using same format as original preprocess (score, joke, body, punchline)
    train_df[["score", "joke", "body", "punchline"]].to_csv(train_path, sep="\t", quoting=csv.QUOTE_NONE, escapechar='\\', index=None, header=None, encoding="UTF-8")
    test_df[["score", "joke", "body", "punchline"]].to_csv(test_path, sep="\t", quoting=csv.QUOTE_NONE, escapechar='\\', index=None, header=None, encoding="UTF-8")
    
    print(f"Saved to {train_path} and {test_path}")

if __name__ == "__main__":
    create_2015_2019_splits()
