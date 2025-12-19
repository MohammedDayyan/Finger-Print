import tensorflow as tf
from config.settings import IMAGE_SIZE
from keras.applications.mobilenet_v2 import preprocess_input

def load_image(path: str) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3,expand_animations=False)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = preprocess_input(img)
    #img = img / 255.0
    return img