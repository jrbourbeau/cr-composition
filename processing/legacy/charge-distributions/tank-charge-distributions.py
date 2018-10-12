
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import glob
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from bokeh.io import output_notebook
from dask import delayed
import dask.array as da
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, ProgressBar, visualize
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import standardize

import keras
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, Cropping2D, Flatten,
                          Dense, Dropout)
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical


import comptools as comp

config = 'IC86.2012'
num_groups = 2

comp_list = comp.get_comp_list(num_groups=num_groups)
energybins = comp.get_energybins(config=config)
log_energy_min = energybins.log_energy_min
log_energy_max = energybins.log_energy_max

df_sim_train, df_sim_test = comp.load_sim(config=config,
                                          log_energy_min=log_energy_min,
                                          log_energy_max=log_energy_max,
                                          test_size=0.5,
                                          verbose=True)

feature_list, feature_labels = comp.get_training_features()
feature_list += ['lap_x', 'lap_y']

X = df_sim_train[feature_list].values
y = df_sim_train['comp_target_{}'.format(num_groups)].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)


# ## Load charge distribution histograms
file_pattern = os.path.join(comp.paths.comp_data_dir,
                            config,
                            'i3_hdf_sim',
                            '*_charge-dist.hdf5')
filenames = glob.glob(file_pattern)

dask_arrays = []
for fn in filenames:
    with h5py.File(fn, mode='r') as f:
        d = f['/charge_dist'].value
#         d = d / d.max()
        d = np.nan_to_num(d)
        d = np.array([i / i.sum() for i in d])
    x = da.from_array(d, chunks=100)
    dask_arrays.append(x)

x = da.concatenate(dask_arrays, axis=0)  # concatenate arrays along first axis
with ProgressBar():
    hists = x.compute(num_workers=20)

print(f'current_shape = {hists.shape}')
new_shape = tuple(list(hists.shape) + [1])
hists = hists.reshape(new_shape)
print(f'new_shape = {hists.shape}')

dask_arrays = []
for fn in filenames:
    with h5py.File(fn, mode='r') as f:
        d = f['/event_id'].value
        # Byte strings
        d = np.array([i.decode('utf-8') for i in d])
    x = da.from_array(d, chunks=100)
    dask_arrays.append(x)

event_ids = da.concatenate(dask_arrays, axis=0)  # concatenate arrays along first axis
with ProgressBar():
    event_ids = event_ids.compute(num_workers=20)

train_mask = np.array([item in df_sim_train.index for item in event_ids])
test_mask = np.array([item in df_sim_test.index for item in event_ids])

hist_array_train = xr.DataArray(hists[train_mask], coords={'dim_0': event_ids[train_mask]})
hist_array_test = xr.DataArray(hists[test_mask], coords={'dim_0': event_ids[test_mask]})

hist_array_train = hist_array_train.loc[df_sim_train.index]
hist_array_test = hist_array_test.loc[df_sim_test.index]

# ## CNN model

y_cat = to_categorical(y)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

inputs = Input(shape=(24, 24, 1), name='hist_input')

x = Conv2D(12, (3, 3), padding='same')(inputs)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(12, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(24, (3, 3), padding='same')(inputs)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(24, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(64, (3, 3), padding='same')(inputs)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=0.3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

flatten_output = Flatten()(x)

auxiliary_input = Input(shape=(len(feature_list),), name='aux_input')
x = keras.layers.concatenate([flatten_output, auxiliary_input])

x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

predictions = Dense(num_groups, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=[inputs, auxiliary_input], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


X_stand = standardize(X)

np.random.seed(2)
n_samples = 6000
hist = model.fit([hist_array_train[:n_samples], X_stand[:n_samples]], y_cat[:n_samples],
                 epochs=10,
                 batch_size=128,
                 validation_split=0.3)
