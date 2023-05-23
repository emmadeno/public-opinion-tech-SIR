import sys
import os
import json
import glob
import time
import pandas as pd
from itertools import product
from multiprocessing import Pool
import subprocess
import  tarfile
import xml.etree.ElementTree as ET
import zipfile
import lib.utils as U
import lib.constants as C
import math

from functools import partial

import spacy
from typing import List
import numpy as np
from joblib import Parallel, delayed
from IPython.display import clear_output
import matplotlib.pyplot as plt
import ast
import random
import json
import operator
import time
import os
import re
pd.options.mode.chained_assignment = None  # default='warn'

import spacy
from typing import List
import tomotopy as tp
import pandas as pd
import numpy as np
import spacy
from scipy.stats import median_abs_deviation
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.base import BaseEstimator
from gensim import corpora
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
from joblib import Parallel, delayed
from IPython.display import clear_output
import wordcloud
import matplotlib.pyplot as plt
from nltk.corpus import reuters

import lib.utils as U
import lib.constants as C
import lib.plots as PL
import lib.processing as PR
import lib.lemmas as SP
import lib.embeddings as EMB

import pandas as pd
import io
import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
from scipy import spatial

import fasttext
import csv

from wmd import WMD
from networkx.algorithms import community
import networkx as nx