from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
import keras

def build_convnet_model(args, last_layer=True, sr=12000, compile=True):
    model = raw_vgg(args, tf=args.tf_type, normalize=args.normalize,
                    decibel=args.decibel, last_layer=last_layer, sr=sr)
    if compile:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-3),
                      loss='binary_crossentropy')
    return model