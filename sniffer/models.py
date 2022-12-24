import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Filter info MSGS

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input, Dropout, GRU, Embedding

def GloVe_embedding_layer(infpath, MAX_SEQUENCE_LENGTH):
    """
    Function to initialize the embedding layer with GloVe
    model weights
    
    Args
    ---------
        infpath: path to txt file with learned vectors
        MAX_SEQUENCE_LENGTH: maximum length of token sequence
    
    Returns
    ---------
        embedding_layer: keras embedding layer with frozen weights
    """
    vocab_size=400001
    embeddings_index = {}

    f = open(infpath)
    for i, line in enumerate(f):
        if i == 400000:
            break
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in embeddings_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                            300,
                                            weights=[embedding_matrix],
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            trainable=False)
    return embedding_layer


def LSTM_model_frozen(MAX_SEQUENCE_LENGTH, embedding_layer):
    """
    LSTM model with frozen embedding weights
    
    Args
    ---------
        MAX_SEQUENCE_LENGTH: maximum length of token sequence
        embedding_layer: keras embedding layer with frozen weights
    
    Returns
    ---------
        model: keras LSTM model with frozen embedding layer
    """
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(embedding_sequences)
    x = Dense(512, activation='relu')(x) # Layers from here are only applied to the output
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(x)
    model = tf.keras.Model(sequence_input, outputs)
    return model

def GRU_model_frozen(MAX_SEQUENCE_LENGTH, embedding_layer):
    """
    GRU model with frozen embedding weights
    
    Args
    ---------
        MAX_SEQUENCE_LENGTH: maximum length of token sequence
        embedding_layer: keras embedding layer with frozen weights
    
    Returns
    ---------
        model: keras GRU model with frozen embedding layer
    """
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_sequences = embedding_layer(sequence_input)
    x = Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2))(embedding_sequences)
    x = Dense(512, activation='relu')(x) # Layers from here are only applied to the output
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(x)
    model = tf.keras.Model(sequence_input, outputs)
    return model

def GRU_model_trainable(MAX_SEQUENCE_LENGTH, vocab_size):
    """
    GRU model with tunable embedding weights
    
    Args
    ---------
        MAX_SEQUENCE_LENGTH: maximum length of token sequence
        vocab_size: size of vocab
    
    Returns
    ---------
        model: keras GRU model with tunable embedding layer
    """
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    x = Embedding(vocab_size, 300, input_length=MAX_SEQUENCE_LENGTH, trainable=True)(sequence_input)
    x = Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(512, activation='relu')(x) # Layers from here are only applied to the output
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(x)
    model = tf.keras.Model(sequence_input, outputs)
    return model

def LSTM_model_trainable(MAX_SEQUENCE_LENGTH, vocab_size):
    """
    LSTM model with tunable embedding weights
    
    Args
    ---------
        MAX_SEQUENCE_LENGTH: maximum length of token sequence
        vocab_size: size of vocab
    
    Returns
    ---------
        model: keras LSTM model with tunable embedding layer
    """
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    x = Embedding(vocab_size, 300, input_length=MAX_SEQUENCE_LENGTH, trainable=True)(sequence_input)
    x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(512, activation='relu')(x) # Layers from here onward are applied only to output
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(x)
    model = tf.keras.Model(sequence_input, outputs)
    return model