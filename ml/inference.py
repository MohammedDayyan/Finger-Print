import tensorflow as tf
import numpy as np
from data.preprocess import load_image
from config.settings import MODEL_PATH

model = tf.keras.models.load_model(MODEL_PATH)
gender_map = {
    1: "Male",
    0: "Female"
}
finger_map = {
    0: "Index",
    1: "Little",
    2: "Middle",
    3: "Ring",
    4: "Thumb"
}
hand_map = {
    0: "Left",
    1: "Right"
}
type_map ={
    0: "Real",
    1: "Altered-easy",
    2: "Altered-medium",
    3: "Altered-Hard"
}
def predict_fingerprint(image_path: str) -> dict:
    img = load_image(image_path)
    img = tf.expand_dims(img, 0)

    pred_type, pred_gender, pred_hand, pred_finger = model.predict(img)

    return {
        "type": {'label':type_map[int(np.argmax(pred_type))],
                 'confidence': float(round(np.max(pred_type),2))
        },
        "gender": {'label':gender_map[int(np.argmax(pred_gender))],
                   'confidence': float(round(np.max(pred_gender),2))
        },
        "hand": {'label':hand_map[int(np.argmax(pred_hand))],
                 'confidence': float(round(np.max(pred_hand),2))
        },
        "finger": {'label':finger_map[int(np.argmax(pred_finger))],
                   'confidence': float(round(np.max(pred_finger),2))
        }
    }