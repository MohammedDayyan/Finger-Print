from data.dataset import build_dataset
from config.settings import CSV_PATH,BATCH_SIZE,MODEL_PATH
import pandas as pd
import keras 

def evaluation():
    df = pd.read_csv(CSV_PATH)
    test_df = df.iloc[int(0.8 * len(df)):]
    test_ds = build_dataset(test_df,BATCH_SIZE)
    
    
    
    model = keras.models.load_model(MODEL_PATH)
    
    results = model.evaluate(test_ds,return_dict=True)
    
    print(results)
    
if __name__ == '__main__':
    evaluation()

