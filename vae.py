import tensorflow as tf
import numpy as np

lstm_dim = 64
max_smiles_len = 100;
latent_dim = 64

SMILES_CHARS = [' ',
                  '#', '%', '(', ')', '+', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '=', '@',
                  'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                  'R', 'S', 'T', 'V', 'X', 'Z',
                  '[', '\\', ']',
                  'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                  't', 'u']

input_dim = (max_smiles_len, len(SMILES_CHARS))
output_dim = (max_smiles_len, len(SMILES_CHARS))


smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))

with open('smiles.txt') as f:
    smiles_as_list = f.readlines()


def smiles_to_onehot(smiles, max_len = 100):
    onehot = np.zeros((max_len, len(SMILES_CHARS)))
    for i, c in enumerate(smiles):
        onehot[i, smi2index[c]] = 1
    return onehot


def smiles_decoder( onehot ):
    smi = ''
    onehot = onehot.argmax( axis=-1 )
    for i in onehot:
        smi += index2smi[i]
    return smi


decoded_rnn_size = 64
encoded_rnn_size = 64
batch_size = 1


def create_network():
    create_encoder()
    create_decoder()


def create_encoder():
    with tf.variable_scope('encode'):
        encode_cell = []
        for i in range(encoded_rnn_size):
            encode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
        global encoder
        encoder = tf.nn.rnn_cell.MultiRNNCell(encode_cell)


def create_decoder():
    with tf.variable_scope('decode'):
        decode_cell = []
        for i in range(decoded_rnn_size):
            decode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
        global decoder
        decoder = tf.nn.rnn_cell.MultiRNNCell(decode_cell)


def encode(X):
    repr, state = tf.nn.dynamic_rnn(encoder, X, dtype=tf.float32, scope='encode', sequence_length=L)
    mu = tf.keras.layers.Dense(latent_dim, activation="relu")(repr)
    sg = tf.keras.layers.Dense(latent_dim, activation=tf.nn.softplus)(repr)
    z =[i+(sg*np.random.uniform(-1.0, 1.0, size=[1])) for i in mu]
    return z


def decode(Z):
    initial_decoded_state=decoder.zero_state(batch_size = batch_size, dtype=tf.float32)
    Y,  output_decoded_state = tf.nn.dynamic_rnn(decoder, Z, dtype=tf.float32, scope='decode', sequence_length=L, initial_state=initial_decoded_state)

    return Y


def calculate_loss():


if __name__ == '__main__':
    create_network()
    X = tf.placeholder(tf.float32, shape=[max_smiles_len,len(SMILES_CHARS)])
    L = tf.placeholder(tf.float32, [batch_size])

    z = encode(X)

    calculate_loss()

    #output = decode(z)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out = sess.run(z, feed_dict={X: smiles_to_onehot("C[C@@]1(C(=O)C=C(O1)C(=O)[O-])c2ccccc2"), L: batch_size})
    print(out)
    #latent_z = encoder(X)
    #output = decoder(latent_z)

