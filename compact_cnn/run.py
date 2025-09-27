import sys
import numpy as np
from compact_cnn.prepare_audio import prepare_audio
import compact_cnn.models as my_models
from argparse import Namespace
from keras import backend as K

# Example tag list (replace with your dataset’s tags if different)
TAGS = [
    "rock", "pop", "alternative", "indie", "electronic", "female vocalists",
    "dance", "00s", "alternative rock", "jazz", "beautiful", "metal",
    "chillout", "male vocalists", "classic rock", "soul", "indie rock",
    "Mellow", "electronica", "80s", "folk", "90s", "chill", "instrumental",
    "punk", "oldies", "blues", "hard rock", "ambient", "acoustic", "experimental",
    "Hip-Hop", "70s", "party", "country", "easy listening", "sexy", "catchy",
    "funk", "electro", "heavy metal", "progressive rock", "60s", "rnb",
    "indie pop", "sad", "house", "happy", "reggae", "classical"
]


def load_model(mode="tagger"):
    """Build and return the convnet model."""
    assert mode in ("feature", "tagger")

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

    model = my_models.build_convnet_model(args, last_layer=(mode == "tagger"))
    return model


def run(filename, top_k=5, threshold=0.2):
    # Prepare audio (already batch-ready: (1, 1, 348000))
    audio = prepare_audio(filename)

    # Ensure channel-first format for Kapre
    assert K.image_data_format() == "channels_first", (
        'Keras must be set to "channels_first". '
        "Check ~/.keras/keras.json"
    )

    # Load model
    model = load_model("tagger")

    # Predict
    preds = model.predict(audio)[0]  # shape: (num_tags,)
    
    # Get top-k predictions above threshold
    top_indices = np.argsort(preds)[::-1][:top_k]
    results = [(TAGS[i], preds[i]) for i in top_indices if preds[i] >= threshold]

    print("\n🎶 Top Predictions:")
    for tag, score in results:
        print(f"{tag:20s} {score:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <audiofile>")
    else:
        filename = sys.argv[1]
        run(filename)
