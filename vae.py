import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, LSTM, Lambda, RepeatVector
from keras.models import Model
from keras import objectives

lstm_dim = 64
max_smiles_len = 100
latent_dim = 64

SMILES_CHARS = [' ',
                  '#', '%', '(', ')', '+', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '=', '@',
                  'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                  'R', 'S', 'T', 'V', 'X', 'Z',
                  '[', '\\', ']',
                  'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                  't', 'u','\n']

input_dim = (max_smiles_len, len(SMILES_CHARS))
output_dim = (max_smiles_len, len(SMILES_CHARS))


smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))

with open('smallsmiles.txt') as f:
    smiles_as_list = f.readlines()


def smiles_to_onehot(smiles, max_len = 100):
    onehot = np.zeros((max_len, len(SMILES_CHARS)))
    for i, c in enumerate(smiles):
        onehot[i, smi2index[c]] = 1
    return onehot


def smiles_decoder(onehot):
    smi = ''
    onehot = onehot.argmax( axis=-1 )
    for i in onehot:
        smi += index2smi[i]
    return smi

decoded_rnn_size = 64
encoded_rnn_size = 64
batch_size = 1


input = Input(shape=input_dim)
lstm = LSTM(latent_dim, activation='relu')(input)
zmean = Dense(latent_dim, name='Z_mean_t')(lstm)
zvar = Dense(latent_dim, name='Z_log_var_t', activation=tf.nn.softplus)(lstm)
z = Lambda(lambda m: m[0] + m[1] * tf.random.normal(tf.shape(m[0])))([zmean, zvar])
encoder = Model(input, z)

latent_inputs = Input(shape=latent_dim, name='z_sampling')
repeated = RepeatVector(100)(latent_inputs)
x_2 = LSTM(57, activation='relu', return_sequences=True)(repeated)
decoder = Model(latent_inputs, x_2)

# h_decoded = RepeatVector(100)(z)
# h_decoded = LSTM(57, return_sequences=True)(h_decoded)
# vae_ = Model(input, h_decoded)


def calculate_loss(x, x_decoded_mean):
    xent_loss = objectives.mse(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + zvar - K.square(zmean) - K.exp(zvar))
    loss = xent_loss + kl_loss
    return loss


if __name__ == '__main__':

    numpy_X = [smiles_to_onehot(x) for x in smiles_as_list]
    numpy_X = np.array(numpy_X)
    X = tf.convert_to_tensor(numpy_X, dtype=tf.float32)

    # vae_.compile(loss=calculate_loss, optimizer='adam')
    # vae_.fit(X, X, steps_per_epoch=100, epochs=1)

    outputs = decoder(encoder(X))
    vae = Model(input, outputs)
    vae.compile(loss=calculate_loss, optimizer='adam')
    vae.fit(X, X, steps_per_epoch=100, epochs=5)

    mu = vae.predict(X, steps=1)
    print(mu)
