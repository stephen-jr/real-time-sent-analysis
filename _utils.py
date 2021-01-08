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
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Activation, Flatten
from keras.metrics import Precision, Recall

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream, API, Cursor


# Generic function definition


def f1_m(pcsn, rcl):
    return 2 * ((pcsn * rcl) / (pcsn + rcl + k.epsilon()))


def root_dir(filepath):
    abs_path = os.path.abspath(os.path.dirname(__file__))
    if filepath:
        return os.path.join(abs_path, filepath)
    else:
        return os.path.abspath(os.path.dirname(__file__))
