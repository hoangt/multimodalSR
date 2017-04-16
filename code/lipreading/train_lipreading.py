from __future__ import print_function

import os
import time
from tqdm import tqdm
import lasagne
import numpy as np
import preprocessLipreading

import logging
logger_train = logging.getLogger('lipreading.train')
logger_train.setLevel(logging.DEBUG)


# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn, val_fn,
          network_output_layer,
          batch_size,
          LR_start, LR_decay,
          num_epochs,
          dataset,
          loadPerSpeaker=False,
          save_path=None,
          shuffleEnabled=True):


    X_train, y_train, X_val, y_val, X_test, y_test = dataset
    logger_train.info("the number of training examples is: %s", len(X_train))
    logger_train.info("the number of valid examples is:    %s", len(X_val))
    logger_train.info("the number of test examples is:     %s", len(X_test))


    # A function which shuffles a dataset
    def shuffle(X, y):
        shuffle_parts = 1
        chunk_size = len(X) / shuffle_parts
        shuffled_range = range(chunk_size)

        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])

        for k in range(shuffle_parts):

            np.random.shuffle(shuffled_range)

            for i in range(chunk_size):
                X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
                y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

            X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
            y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

        return X, y


    # This function runs the model a full epoch (on the whole dataset)
    # is a LR is specified, the model will be trained, and the ouput is 'cost' (no accuracy for speed)
    # if no LR is specified, there's not training and the output is 'cost, accuracy'
    def run_epoch(X, y, LR=None):
        cost = 0;     cst = 0;
        accuracy = 0; acc = 0;
        nb_batches = len(X) / batch_size

        predictions = []  # only used if get_predictions = True
        for i in tqdm(range(nb_batches), total=nb_batches):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            # logger_train.info("batch_X.shape: %s", batch_X.shape)
            # logger_train.info("batch_y.shape: %s", batch_y.shape)
            if LR != None:   cst = train_fn(batch_X, batch_y, LR)      # training
            else:            cst, acc = val_fn(batch_X, batch_y)  # validation
            cost += cst;    accuracy += acc

        cost /= nb_batches;
        accuracy /= nb_batches

        if LR != None: return cost, nb_batches  # for training, only cost (faster)
        else:          return cost, accuracy, nb_batches  # for validation, get both

    best_val_err = 100
    best_epoch = 1
    LR = LR_start

    # We iterate over epochs:
    logger_train.info("starting training for %s epochs...", num_epochs)
    for epoch in range(num_epochs):
        logger_train.info("\n\n\n Epoch %s started", epoch + 1)
        start_time = time.time()

        train_loss, _ = run_epoch(X=X_train, y=y_train, LR=LR)
        X_train, y_train = shuffle(X_train, y_train)

        val_err, val_loss, _ = run_epoch(X=X_val, y=y_val)

        # test if validation error went down
        printTest = False
        if val_err <= best_val_err:
            printTest = True
            best_val_err = val_err
            best_epoch = epoch + 1

            logger_train.info("Best ever validation score; evaluating TEST set...")

            if not loadPerSpeaker:  #all at once
                test_err, test_loss, _ = run_epoch(X_test, y_test)


            logger_train.info("TEST results: ")
            logger_train.info("\t  test loss:                     %s", str(test_loss))
            logger_train.info("\t  test error rate:               %s %%", str(test_err))

            if save_path is None:
                save_path = "./bestModel"
            if not os.path.exists(os.dirname(save_path)):
                os.makedirs(os.dirname(save_path))
            logger_train.info("saving model to %s", save_path)
            np.savez(save_path, *lasagne.layers.get_all_param_values(network_output_layer))

        epoch_duration = time.time() - start_time

        # Then we logger_train.info the results for this epoch:
        logger_train.info("Epoch %s of %s took %s seconds", epoch + 1, num_epochs, epoch_duration)
        logger_train.info("  LR:                            %s",    str(LR))
        logger_train.info("  training loss:                 %s",    str(train_loss))
        logger_train.info("  validation loss:               %s",    str(val_loss))
        logger_train.info("  validation error rate:         %s %%", str(val_err))
        logger_train.info("  best epoch:                    %s",    str(best_epoch))
        logger_train.info("  best validation error rate:    %s %%", str(best_val_err))
        if printTest:
            logger_train.info("  test loss:                     %s",    str(test_loss))
            logger_train.info("  test error rate:               %s %%", str(test_err))

        # decay the LR
        LR *= LR_decay

    logger_train.info("Done.")
