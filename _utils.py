# Import libraries
import os
import re
import csv
import sys
import dill
import time
import pandas as pd
import numpy as np


from keras import backend as k
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream, API, Cursor


# Generic function definition
def rcll(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())
    return recall


def prcsn(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    return precision


def f1_m(y_true, y_pred):
    pcsn = prcsn(y_true, y_pred)
    rcl = rcll(y_true, y_pred)
    return 2 * ((pcsn * rcl) / (pcsn + rcl + k.epsilon()))
