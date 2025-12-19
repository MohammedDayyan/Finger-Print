import tensorflow as tf

IMAGE_SIZE = 224
BATCH_SIZE = 32

AUTOTUNE = tf.data.experimental.AUTOTUNE 

CSV_PATH = 'data.csv'
MODEL_PATH = 'model/fingerprint_model.h5'