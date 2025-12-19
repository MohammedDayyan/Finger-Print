import tensorflow as tf
from data.loader import load_dataframe
from data.preprocess import load_image
from data.dataset import build_dataset
from ml.model import build_fingerprint_model
from config.settings import *

def train():
    df = load_dataframe(CSV_PATH)
    
    train_ds = build_dataset(
        df=df,
        batch_size=BATCH_SIZE
    )
    
    model = build_fingerprint_model(
        df["alter_lbl"].nunique(),
        df["gender_lbl"].nunique(),
        df["hand_lbl"].nunique(),
        df["finger_lbl"].nunique()
    )

    model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        "type": "sparse_categorical_crossentropy",
        "gender": "sparse_categorical_crossentropy",
        "hand": "sparse_categorical_crossentropy",
        "finger": "sparse_categorical_crossentropy"
    },
    metrics= {
        "type": ["accuracy"],
        "gender": ["accuracy"],
        "hand": ["accuracy"],
        "finger": ["accuracy"]
    }
)
    #for x, y in train_ds.take(1):
     #   print(x.numpy().min(),x.numpy().max())
        
 
    model.fit(train_ds,epochs=5)
    model.save(MODEL_PATH)
    
if __name__ == "__main__" :
    train() 