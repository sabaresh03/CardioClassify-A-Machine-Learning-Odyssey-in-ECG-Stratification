import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import signal




df = pd.read_csv("/content/drive/MyDrive/XProject/heartbeat/ptbdb_abnormal _test.csv",index_col = False,header=None)
rowcount = len(df.index)
colcount = len(df.columns)
print('Number of Rows in dataframe : ' ,rowcount,colcount)


for i in range(colcount):
    high = 20/(1000/2)
    low = 450/(1000/2)
    b,a = sp.signal.butter(4, [high,low], btype='bandpass')
    emg_filtered = sp.signal.filtfilt(b, a, df[i])
    emg_rectified = abs(emg_filtered)
    plt.plot(emg_filtered)
    sd = "/content/drive/MyDrive/XProject/dataset/test_set/abnormal"+"testa"+str(i)+".png"
    plt.savefig(sd)
    plt.show()


dg = pd.read_csv("/content/drive/MyDrive/XProject/heartbeat/ptbdb_abnormal.csv",index_col = False,header=None)
rowcount1 = len(dg.index)
colcount1 = len(dg.columns)
print('Number of Rows in dataframe : ' ,rowcount1,colcount1)

for k in range(colcount1):
    high = 20/(1000/2)
    low = 450/(1000/2)
    b,a = sp.signal.butter(4, [high,low], btype='bandpass')
    emg_filtered1 = sp.signal.filtfilt(b, a, dg[k])
    emg_rectified1 = abs(emg_filtered1)
    plt.plot(emg_filtered1)
    sd1 = "/content/drive/MyDrive/XProject/dataset/test_set/normal"+"testn"+str(k)+".png"
    plt.savefig(sd1)
    plt.show()

df = pd.read_csv("/content/drive/MyDrive/XProject/heartbeat/ptbdb_abnormal _test.csv",index_col = False,header=None)
rowcount = len(df.index)
colcount = len(df.columns)
print('Number of Rows in dataframe : ' ,rowcount,colcount)
#-------------------------------------------------------------------------------------------------

df = pd.read_csv("/content/drive/MyDrive/XProject/heartbeat/ptbdb_abnormal.csv",index_col = False,header=None)
rowcount2 = len(df.index)
colcount2 = len(df.columns)

print('Number of Rows in dataframe : ' ,rowcount2,colcount2)
for i in range(colcount2):
    high = 20/(1000/2)
    low = 450/(1000/2)
    b,a = sp.signal.butter(4, [high,low], btype='bandpass')
    emg_filtered = sp.signal.filtfilt(b, a, df[i])
    emg_rectified = abs(emg_filtered)
    plt.plot(emg_filtered)
    sd = "/content/drive/MyDrive/XProject/dataset/training_set/abnormal"+"traina"+str(i)+".png"
    plt.savefig(sd)
    plt.show()


dg = pd.read_csv("/content/drive/MyDrive/XProject/heartbeat/ptbdb_normal.csv",index_col = False,header=None)
rowcount1 = len(dg.index)
colcount1 = len(dg.columns)
print('Number of Rows in dataframe : ' ,rowcount1,colcount1)

for k in range(colcount1):
    high = 20/(1000/2)
    low = 450/(1000/2)
    b,a = sp.signal.butter(4, [high,low], btype='bandpass')
    emg_filtered1 = sp.signal.filtfilt(b, a, dg[k])
    emg_rectified1 = abs(emg_filtered1)
    plt.plot(emg_filtered1)
    sd1 = "/content/drive/MyDrive/XProject/dataset/test_set/normal"+"trainn"+str(k)+".png"
    plt.savefig(sd1)
    plt.show()

    
-------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
from collections import defaultdict
# This dictionary holds a mapping {graph: UID_DICT}.
# each UID_DICT is a dictionary mapping name prefixes to a current index,
# used for generatic graph-specific string UIDs
# for various names (e.g. layer names).
_GRAPH_UID_DICTS = {}

# This boolean flag can be set to True to leave variable initialization
# up to the user.
# Change its value via `manual_variable_initialization(value)`.
_MANUAL_VAR_INIT = False


def get_uid(prefix=''):
    global _GRAPH_UID_DICTS
    graph = tf.get_default_graph()
    if graph not in _GRAPH_UID_DICTS:
        _GRAPH_UID_DICTS[graph] = defaultdict(int)
    _GRAPH_UID_DICTS[graph][prefix] += 1
    return _GRAPH_UID_DICTS[graph][prefix]


def reset_uids():
    global _GRAPH_UID_DICTS
    _GRAPH_UID_DICTS = {}




# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])






# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/XProject/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/XProject/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=376,  # Number of batches per epoch
    epochs=20,
    validation_data=test_set,
    validation_steps=190  # Number of batches for validation
)





from keras.preprocessing import image
test_image = image.load_img('/content/drive/MyDrive/XProject/random.png',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
print(result[0][0])
if result[0][0]>=0.5:
 prediction='normal'
else:
 prediction = 'abnormal'
print(prediction)
