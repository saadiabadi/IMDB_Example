from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D
import random

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

model_arch = {
    'Embedding': [max_features, embedding_size, maxlen],
    # 'Dropout': [0.25],
    'Conv1D': [filters, kernel_size,'valid', 'relu', 1],
    # 'MaxPooling1D': [pool_size],
    'LSTM': [lstm_output_size],
    'Dense': [1, 'sigmoid']
}


def create_seed_model(trainedLayers=0):
    """
    Helper function to generate initial seed model.
    Define CNN-LSTM architecture
    :return: model
    """

    lay_count = 0
    ######### start building the model
    model = Sequential()

    if trainedLayers > 0:

        randomlist = random.sample(range(0, len(model_arch)), trainedLayers)
        print(randomlist, flush=True)

        with open('/app/layers.txt', '+a') as f:
            print(randomlist, file=f)

        for key, item in model_arch.items():
            if lay_count in randomlist:
                if key in 'Embedding':
                    model.add(Embedding(item[0], item[1], input_length=item[2], trainable=True))
                    model.add(Dropout(0.25))
                elif key in 'Conv1D':
                    model.add(Conv1D(item[0], item[1], padding=item[2],
                                     activation=item[3],strides=item[4],
                                     trainable=True))
                    model.add(MaxPooling1D(pool_size=pool_size))
                elif key in 'LSTM':
                    model.add(LSTM(item[0], trainable=True))
                else:
                    model.add(Dense(item[0], activation=item[1], trainable=True))

            else:
                if key in 'Embedding':
                    model.add(Embedding(item[0], item[1], input_length=item[2], trainable=False))
                    model.add(Dropout(0.25))
                elif key in 'Conv1D':
                    model.add(Conv1D(item[0], item[1], padding=item[2],
                                     activation=item[3], strides=item[4],
                                     trainable=False))
                    model.add(MaxPooling1D(pool_size=pool_size))
                elif key in 'LSTM':
                    model.add(LSTM(item[0], trainable=False))
                else:
                    model.add(Dense(item[0], activation=item[1], trainable=False))

            lay_count += 1

        print(" --------------------------------------- ", flush=True)
        print(" ------------------Partial MODEL CREATED------------------ ", flush=True)
        print(" --------------------------------------- ", flush=True)
    else:

        model.add(Embedding(max_features, embedding_size, input_length=maxlen))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(lstm_output_size))
        model.add(Dense(1, activation='sigmoid'))

        print(" --------------------------------------- ", flush=True)
        print(" ------------------Full MODEL CREATED------------------ ", flush=True)
        print(" --------------------------------------- ", flush=True)

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model




