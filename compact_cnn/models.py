# compact_cnn/models.py
"""
Compact CNN for music auto-tagging.
Input: mel-spectrogram computed by librosa, shape (n_mels, time, 1).
Compatible with TF 2.20 + Keras 3.x (no kapre required).
"""

from types import SimpleNamespace
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, Flatten, Dense
)


def build_convnet_model(args: SimpleNamespace, last_layer: bool = False) -> Model:
    """
    Build a compact CNN for music auto-tagging.
    args must contain: n_mels (int)
    We accept variable time dimension via None.
    """
    n_mels = args.n_mels

    # Input is channels-last: (n_mels, time, 1)
    input_shape = (n_mels, None, 1)
    inp = Input(shape=input_shape, name="mel_input")

    x = Conv2D(64, (3, 3), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)

    if last_layer:
        out = Dense(50, name="logits")(x)
    else:
        out = Dense(50, activation="sigmoid", name="preds")(x)

    model = Model(inputs=inp, outputs=out, name="compact_cnn_mel")
    return model
