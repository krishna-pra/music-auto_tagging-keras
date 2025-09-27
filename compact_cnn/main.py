from keras import backend as K
import models as my_models
from argparse import Namespace
from prepare_audio import prepare_audio
import numpy as np
if __name__ == "__main__":
    main("tagger")

def main(mode, conv_until=None):
    assert mode in ('feature', 'tagger')

    # Use Keras 2.x API
    assert K.image_data_format() == 'channels_first', (
        'image_data_format should be "channels_first". '
        'Set it in ~/.keras/keras.json'
    )

    args = Namespace(
        tf_type='melgram',
        normalize='no',
        decibel=True,
        fmin=0.0, fmax=6000,
        n_mels=96,
        trainable_fb=False,
        trainable_kernel=False,
    )

    model = my_models.build_convnet_model(args, last_layer=(mode == 'tagger'))

    # Example usage
    audio = prepare_audio("example.mp3")
    preds = model.predict(audio[np.newaxis, :, :])
    print(preds)
