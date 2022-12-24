import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Filter info MSGS

import tensorflow as tf
import pandas as pd
import numpy as np
import random

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def seed_everything(seed):
    """
    Function to set seed for reproducability
    """
    np.random.seed(seed)
    tf.random.set_seed(seed) 
    random.seed(seed)

def load_data(infpath):
    """
    Function to initialize the embedding layer with GloVe
    model weights
    
    Args
    ---------
        infpath: path to folder with train.txt, valid.txt and test.txt data
    
    Returns
    ---------
        train: pandas df with text and label pairs
        valid: pandas df with text and label pairs
        test: pandas df with text and label pairs
    """
    # Test Data
    data = []
    with open("{}/test.txt".format(infpath), "r") as f:
        for line in f.readlines():
            data.append([line[2:].strip(), int(line[0])])       
    test = pd.DataFrame(data, columns = ['text', 'label'])

    # Validation Data
    data = []
    with open("{}/valid.txt".format(infpath), "r") as f:
        for line in f.readlines():
            data.append([line[2:].strip(), int(line[0])])        
    valid = pd.DataFrame(data, columns = ['text', 'label'])

    # Training Data
    data = []
    with open("{}/train.txt".format(infpath), "r") as f:
        for line in f.readlines():
            data.append([line[2:].strip(), int(line[0])])  
    train = pd.DataFrame(data, columns = ['text', 'label'])

    print("Train Len: ", len(train))
    print("Valid Len: ", len(valid))
    print("Test Len: ", len(test))

    return train, valid, test

def ohe_labels(train, valid, test, N_LABELS):
    """
    One hot encode labels
    
    Args
    ---------
        train: pandas df with text and label pairs
        valid: pandas df with text and label pairs
        test: pandas df with text and label pairs
        N_LABELS: number of labels in the data
    
    Returns
    ---------
        y_train_raw: numpy array of raw labels
        y_valid_raw: numpy array of raw labels
        y_test_raw: numpy array of raw labels
        y_train: numpy array of encoded labels
        y_valid: numpy array of encoded labels
        y_test: numpy array of encoded labels
    """
    y_train_raw = train['label'].to_numpy()
    y_train = np.eye(N_LABELS)[y_train_raw]

    y_valid_raw = valid['label'].to_numpy()
    y_valid = np.eye(N_LABELS)[y_valid_raw]

    y_test_raw = test['label'].to_numpy()
    y_test = np.eye(N_LABELS)[y_test_raw]

    return y_train, y_valid, y_test

def keras_train(model, x_train, y_train, x_valid, y_valid):
    """
    Main function to train LSTM and GRU models

    Args
    ---------
        model: keras model that will be trained
        x_train: numpy array of tokenized text input
        y_train: numpy array of encoded labels
        x_valid: numpy array of tokenized text input
        y_valid: numpy array of encoded labels
    """

    LR = 1e-3
    BATCH_SIZE = 1024
    EPOCHS = 10
    CHECKPOINTS_FPATH = "./checkpoints/"
    
    model.compile(optimizer=Adam(learning_rate=LR), loss="categorical_crossentropy", metrics=['accuracy'])

    early_stopper_callback = EarlyStopping(
		monitor="val_accuracy",
		min_delta=0,
		patience=3,
		verbose=1
	)

    model_checkpoint_callback = ModelCheckpoint(
		filepath = CHECKPOINTS_FPATH,
		save_weights_only=True,
		monitor='val_accuracy',
		mode='max',
		save_best_only=True)

    if tf.config.experimental.list_physical_devices('GPU'):
        print("Training on GPU...") 
    else:
        print("Training on CPU...")

    history = model.fit(
        x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=(x_valid, y_valid), callbacks=[early_stopper_callback, model_checkpoint_callback],
        shuffle=True)

    # Restoring best weights and saving model
    model.load_weights(CHECKPOINTS_FPATH)
    model.save("best_model")

def evaluate(model, x_train, x_valid, x_test, y_train, y_valid, y_test):
    """
    Function to evaluate the Cateogorical Cross Entropy
    and accuracy of the model

    Args
    ---------
        model: keras model that will be trained
        x_train, x_valid, x_test: numpy array of tokenized text input
        y_train, y_valid, y_test: numpy array of encoded labels
    """

    # Mapping of label encoder
    label_dict = {
        0:"Substitution",
        1:"Transposition",
        2:"Reverse",
        3:"Shift",
        4:"Wordflip",
        5:"Original",
    }

    # Converting OHE labels to Actual Labels
    y_train_raw = np.where(y_train==1)[1]
    y_valid_raw = np.where(y_valid==1)[1]
    y_test_raw = np.where(y_test==1)[1]

    # Setting evaluation metrics
    cce = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.Accuracy()

    # Training 
    preds = model.predict(x_train)
    print("Train CCE: {:.4f}".format(1 - cce(y_train, preds).numpy()))
    print("Train ACC: {:.4f}".format(acc(y_train_raw, preds.argmax(1)).numpy()))

    # Validation
    preds = model.predict(x_valid)
    print("Valid CCE: {:.4f}".format(1 - cce(y_valid, preds).numpy()))
    print("Valid ACC: {:.4f}".format(acc(y_valid_raw, preds.argmax(1)).numpy()))

    # Testing
    preds = model.predict(x_test)
    print("Test CCE: {:.4f}".format(1 - cce(y_test, preds).numpy()))
    print("Test ACC: {:.4f}".format(acc(y_test_raw, preds.argmax(1)).numpy()))

    # Getting ACC by class on Test Data
    print("\n --- By class scores --- \n")
    correct = y_test_raw[(y_test_raw == preds.argmax(1))]
    correct_totals = np.zeros(6, dtype=int)
    for val in correct:
        correct_totals[val]+=1
    _, totals = np.unique(y_test_raw, return_counts=True)

    # Printing Results
    for class_type, score in zip(label_dict.values(), correct_totals/totals):
        print("{}: {:.4f}".format(class_type, score))
    return