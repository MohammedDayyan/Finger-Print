import tensorflow as tf
from data.preprocess import load_image

def build_dataset(df,  batch_size, shuffle=True):
    image_paths = df["path"].values
    labels = {
        "type": df["alter_lbl"].values,
        "gender": df["gender_lbl"].values,
        "hand": df["hand_lbl"].values,
        "finger": df["finger_lbl"].values
    }

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    def _map(path, lbls):
        img = load_image(path)
        return img, lbls

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds