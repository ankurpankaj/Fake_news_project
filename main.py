##import tensorflow as tf
##from tensorflow.python.keras.preprocessing.text import Tokenizer
##from tensorflow.python.keras.models import Sequential
##from tensorflow.python.keras.optimizers import Adam
##from tensorflow.python.keras import regularizers
#import tensorflow.python.keras.utils as ku
import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.model_selection import train_test_split

import pickle

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

cl_fake = list(df_fake.columns)
#df_fake[cl_fake[0]] = df_fake[cl_fake[0]].str.cat(df_fake[cl_fake[1]], sep = " ")

cl_true = list(df_true.columns)
#df_true[cl_true[0]] = df_true[cl_true[0]].str.cat(df_true[cl_true[1]], sep = " ")

data = {'news':[], 'label':[]}

for n,i in enumerate(df_fake[cl_fake[0]]):
    data['news'].append(i+" "+df_fake[cl_fake[1]][n])
    data['label'].append(0)

for n,i in enumerate(df_true[cl_true[0]]):
    data['news'].append(i+" "+df_true[cl_true[1]][n])
    data['label'].append(1)

data = pd.DataFrame(data)
data = utils.shuffle(data)

train, test = train_test_split(data, test_size=0.3, random_state=42)

pickle_out = open("train.pickle","wb")
pickle.dump(train, pickle_out)
pickle_out.close()

pickle_out = open("test.pickle","wb")
pickle.dump(test, pickle_out)
pickle_out.close()
