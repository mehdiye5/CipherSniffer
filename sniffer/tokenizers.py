import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Filter info MSGS

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import string
import numpy as np

"""
Scripts to train tokenizer models.
"""

def bpe_train(infpath, outfpath):
    """
    Function to train and save byte-pair encoding model.
    
    Args
    ---------
        infpath: path to input txt file
        outfpath: folder where tokenizer weights are saved
    """
    bpe_tokenizer = Tokenizer(BPE())

    bpe_trainer = BpeTrainer(
        special_tokens = ["[UNK]"],
        vocab_size=10000,
        min_frequency=5,
        show_progress=True
        )

    bpe_tokenizer.pre_tokenizer = Whitespace()
    bpe_tokenizer.train(files=[infpath], trainer=bpe_trainer)
    bpe_tokenizer.save("{}/BPE_trained.json".format(outfpath))
    return
    
def wordpiece_train(infpath, outfpath):
    """
    Function to train and save wordpiece tokenization model.

    Args
    ---------
        infpath: path to input txt file
        outfpath: folder where tokenizer weights are saved
    """
    wp_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    wp_trainer = WordPieceTrainer(
        special_tokens = ["[UNK]"],
        vocab_size=10000,
        min_frequency=5,
        show_progress=True
        )

    wp_tokenizer.pre_tokenizer = Whitespace()
    wp_tokenizer.train(files=[infpath], trainer=wp_trainer)
    wp_tokenizer.save("{}/WP_trained.json".format(outfpath))
    return
    
def tokenizer_train(infpath, outfpath):
    """
    Function to train and save a keras word level tokenizer w/ pickle.
    
    Args
    ---------
        infpath: path to input txt file
        outfpath: folder where tokenizer weights are saved
    """
    data = []
    with open(infpath, "r") as f:
        for line in f.readlines():
            data.append(line.strip())
            
    tokenizer = text.Tokenizer(num_words=400000)
    tokenizer.fit_on_texts(data)

    with open('{}/tokenizer.pickle'.format(outfpath), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def character_level(train, valid, test, MAX_SEQUENCE_LENGTH):
    """
    Applies character-level tokenization to text column in dataframe
    
    Args
    ---------
        train: pandas df with text and label pairs
        valid: pandas df with text and label pairs
        test: pandas df with text and label pairs
        MAX_SEQUENCE_LENGTH: maximum length of token sequence

    Returns
    ---------
        x_train: numpy array of tokenized input text
        x_valid: numpy array of tokenized input text
        x_test: numpy array of tokenized input text
        vocab_size: size of vocab
    """
    x_train = np.array(train['text'].apply(lambda x: string_to_list(x)).values.tolist())
    x_valid = np.array(valid['text'].apply(lambda x: string_to_list(x)).values.tolist())
    x_test = np.array(test['text'].apply(lambda x: string_to_list(x)).values.tolist())
    vocab_size = 28
    return x_train, x_valid, x_test, vocab_size

def string_to_list(x, MAX_SEQUENCE_LENGTH):
    """
    Helper function for the character-level tokenizer
    
    Args
    ---------
        x: pandas column of input text
        MAX_SEQUENCE_LENGTH: maximum length of token sequence
    """
    # All possible characters
    hm = {}
    for i, val in enumerate(string.ascii_lowercase + " "):
        hm[val] = i

    # Pad start of array with OOV token (27)
    arr = [int(hm[x])for x in list(x)]
    arr = [27]*(MAX_SEQUENCE_LENGTH - len(arr)) + arr
    return arr

def subword_level(train, valid, test, tokenizer_file, MAX_SEQUENCE_LENGTH):
    """
    Applies character-level tokenization to text column in dataframe (BPE / Wordpiece)
    
    Args
    ---------
        train: pandas df with text and label pairs
        valid: pandas df with text and label pairs
        test: pandas df with text and label pairs
        tokenizer_file: fpath to the trained tokenizer weights
        MAX_SEQUENCE_LENGTH: maximum length of token sequence

    Returns
    ---------
        x_train: numpy array of tokenized input text
        x_valid: numpy array of tokenized input text
        x_test: numpy array of tokenized input text
        vocab_size: size of vocab
    """
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_file)

    # Apply tokenization
    x_train = pad_sequences([x.ids for x in tokenizer.encode_batch(train['text'].values)],
                            maxlen = MAX_SEQUENCE_LENGTH)
    x_valid = pad_sequences([x.ids for x in tokenizer.encode_batch(valid['text'].values)],
                        maxlen = MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences([x.ids for x in tokenizer.encode_batch(test['text'].values)],
                        maxlen = MAX_SEQUENCE_LENGTH)
    vocab_size = tokenizer.get_vocab_size()
    return x_train, x_valid, x_test, vocab_size

def word_level(train, valid, test, tokenizer_file, MAX_SEQUENCE_LENGTH):
    """
    Applies word-level tokenization to text column in dataframe
    
    Modified from Keras Documentation: 
    https://keras.io/examples/nlp/pretrained_word_embeddings/
    """
    # Loading trained tokenizer
    with open(tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Capping vocab size at 400k
    word_index = {}
    for word, index in tokenizer.word_index.items():
        if index > 400000:
            break
        word_index[index] = word
    vocab_size = len(word_index) + 1

    # Apply tokenization
    x_train = pad_sequences(tokenizer.texts_to_sequences(train['text']),
                            maxlen = MAX_SEQUENCE_LENGTH)
    x_valid = pad_sequences(tokenizer.texts_to_sequences(valid['text']),
                        maxlen = MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(test['text']),
                        maxlen = MAX_SEQUENCE_LENGTH)

    return x_train, x_valid, x_test, vocab_size