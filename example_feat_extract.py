import sys
import numpy as np
from tensorflow.keras.models import Model
from compact_cnn.prepare_audio import prepare_audio
from compact_cnn import models as my_models
from argparse import Namespace
from keras import backend as K


def load_feature_model():
    """Build CNN up to penultimate layer for feature extraction."""
    args = Namespace(
        tf_type="melgram",
        normalize="no",
        decibel=True,
        fmin=0.0,
        fmax=6000,
        n_mels=96,
        trainable_fb=False,
        trainable_kernel=False,
    )
    model = my_models.build_convnet_model(args, last_layer=True)

    # Take output from penultimate dense layer (before sigmoid)
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return feature_model


def run(filename):
    audio = prepare_audio(filename)

    # Ensure channels_first
    assert K.image_data_format() == "channels_first"

    model = load_feature_model()
    features = model.predict(audio)

    print("Extracted feature shape:", features.shape)
    print("Feature vector:", features[0][:10], "...")  # show first 10 values


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_feat_extract.py <audiofile>")
    else:
        run(sys.argv[1])
