import pandas as pd

def load_dataframe(csv_path : str) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path)
    
    df['alter_label'] = df['type'].astype("category").cat.codes 
    df["gender_lbl"] = df["gender"].astype("category").cat.codes
    df["hand_lbl"] = df["hand"].astype("category").cat.codes
    df["finger_lbl"] = df["index"].astype("category").cat.codes

    return df