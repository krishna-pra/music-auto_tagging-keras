"""
models.py
Defines CNN model architectures for music auto-tagging.
Updated for TensorFlow 2.x + Kapre ≥0.3.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Activation, Dropout, Flatten,
    Convolution2D, MaxPooling2D, BatchNormalization
)
from kapre import Melspectrogram, Magnitude

try:
    from kapre.augmentation import AdditiveNoise
except ImportError:
    AdditiveNoise = None


def build_convnet_model(args, last_layer=False):
    """
    Build a compact CNN for music auto-tagging.
    Args:
        args: Namespace with spectrogram config.
        last_layer: If True, returns logits (no sigmoid).
    """
    input_shape = (1, args.n_mels, None)  # channels_first format

    # Input layer
    melgram_input = Input(shape=input_shape)

    # Mel-spectrogram frontend (Kapre)
    x = Melspectrogram(
        sr=12000,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        power_melgram=2.0,
        return_decibel_melgram=args.decibel,
        trainable_fb=args.trainable_fb,
        trainable_kernel=args.trainable_kernel,
        input_data_format="channels_first",
        output_data_format="channels_first",
    )(melgram_input)

    # Optional magnitude layer
    x = Magnitude()(x)

    # --- Conv block 1 ---
    x = Convolution2D(64, (3, 3), padding="same", data_format="channels_first")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(x)
    x = Dropout(0.25)(x)

    # --- Conv block 2 ---
    x = Convolution2D(128, (3, 3), padding="same", data_format="channels_first")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(x)
    x = Dropout(0.25)(x)

    # --- Conv block 3 ---
    x = Convolution2D(256, (3, 3), padding="same", data_format="channels_first")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(x)
    x = Dropout(0.25)(x)

    # Flatten + Dense
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)

    # Output layer
    if last_layer:
        output = Dense(50)(x)  # logits
    else:
        output = Dense(50, activation="sigmoid")(x)  # multi-label classification

    model = Model(inputs=melgram_input, outputs=output)
    return model
