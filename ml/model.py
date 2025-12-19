import tensorflow as tf
from keras import layers,models 

def build_fingerprint_model(num_types, num_genders, num_hands, num_fingers):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base.trainable = True
    
    for layer in base.layers[:-40]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)

    type_out = layers.Dense(num_types, activation="softmax", name="type")(x)
    gender_out = layers.Dense(num_genders, activation="softmax", name="gender")(x)
    hand_out = layers.Dense(num_hands, activation="softmax", name="hand")(x)
    finger_out = layers.Dense(num_fingers, activation="softmax", name="finger")(x)

    model = models.Model(
        inputs=base.input,
        outputs=[type_out, gender_out, hand_out, finger_out]
    )

    return model

