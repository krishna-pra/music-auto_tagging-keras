from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GRU, TimeDistributed, Dense, Dropout, Flatten


def music_tagger_crnn(weights=None):
    """
    CRNN (CNN + RNN) for music auto-tagging.
    Input format: (batch, channels=1, freq=96, time=1366) → channels_first
    """
    model = Sequential()

    # Convolution layers (channels_first)
    model.add(Conv2D(64, (3, 3), activation="relu",
                     input_shape=(1, 96, 1366),
                     data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(128, (3, 3), activation="relu", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(128, (3, 3), activation="relu", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    # Flatten → Dense (for now; real CRNN would reshape to feed GRU)
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="sigmoid"))

    if weights:
        model.load_weights(weights)

    return model
