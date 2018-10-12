from __future__ import division
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Want to use tensorflow backend
# See https://keras.io/backend/ for more info on setting the backend for keras
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

import comptools as comp

color_dict = comp.get_color_dict()

config = 'IC79.2010'
num_groups = 4
comp_list = comp.get_comp_list(num_groups)
energybins = comp.get_energybins(config=config)

df_sim_train, df_sim_test = comp.load_sim(config=config, test_size=0.5,
                                          log_energy_min=energybins.log_energy_min,
                                          log_energy_max=energybins.log_energy_max,
                                          verbose=True)

ldf_cols = [col for col in df_sim_train.columns if 'ldf' in col]

isnull_mask_train = df_sim_train[ldf_cols].isnull().sum(axis=1).astype(bool)
isnull_mask_test = df_sim_test[ldf_cols].isnull().sum(axis=1).astype(bool)
zero_ldf = df_sim_train[ldf_cols].sum(axis=1) == 0

X_train = df_sim_train.loc[~isnull_mask_train, ldf_cols].values
X_train = X_train / X_train.sum(axis=1)[:, None]
y_train = df_sim_train.loc[~isnull_mask_train, f'comp_target_{num_groups}'].values

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=2)

X_test = df_sim_test.loc[~isnull_mask_test, ldf_cols].values
X_test = X_test / X_test.sum(axis=1)[:, None]
y_test = df_sim_test.loc[~isnull_mask_test, f'comp_target_{num_groups}'].values

y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)


def get_model(verbose=False):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
    model.add(Dense(num_groups, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    if verbose:
        model.summary()

    return model


get_model(verbose=True)

epochs = 100
batch_size = 200

skf = StratifiedKFold(n_splits=10, random_state=2)
cv_results = defaultdict(list)
for idx, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    print(f'On fold {idx}...')
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train_cat[train_index], y_train_cat[test_index]

    model = get_model()
    history = model.fit(X_train_fold, y_train_fold,
                        epochs=epochs,
                        validation_data=(X_test_fold, y_test_fold),
                        batch_size=batch_size,
                        verbose=0)

    cv_results['acc'].append(history.history['acc'])
    cv_results['val_acc'].append(history.history['val_acc'])

    cv_results['loss'].append(history.history['loss'])
    cv_results['val_loss'].append(history.history['val_loss'])


cv_df = pd.DataFrame()
cv_df['acc_mean'] = np.mean(cv_results['acc'], axis=0)
cv_df['acc_err'] = np.std(cv_results['acc'], axis=0)

cv_df['val_acc_mean'] = np.mean(cv_results['val_acc'], axis=0)
cv_df['val_acc_err'] = np.std(cv_results['val_acc'], axis=0)

cv_df['loss_mean'] = np.mean(cv_results['loss'], axis=0)
cv_df['loss_err'] = np.std(cv_results['loss'], axis=0)

cv_df['val_loss_mean'] = np.mean(cv_results['val_loss'], axis=0)
cv_df['val_loss_err'] = np.std(cv_results['val_loss'], axis=0)



fig, ax = plt.subplots()

ax.plot(range(1, epochs+1), cv_df['acc_mean'], marker='None', color='C0',
        label='Training')
ax.fill_between(range(1, epochs+1),
                cv_df['acc_mean'] + cv_df['acc_err'],
                cv_df['acc_mean'] - cv_df['acc_err'],
                color='C0', alpha=0.25)

ax.plot(range(1, epochs+1), cv_df['val_acc_mean'], marker='None', color='C1',
        label='Validation')
ax.fill_between(range(1, epochs+1),
                cv_df['val_acc_mean'] + cv_df['val_acc_err'],
                cv_df['val_acc_mean'] - cv_df['val_acc_err'],
                color='C1', alpha=0.25)

ax_loss = ax.twinx()
ax_loss.plot(range(1, epochs+1), cv_df['loss_mean'], marker='None', color='C2',
             label='Training')
ax_loss.fill_between(range(1, epochs+1),
                     cv_df['loss_mean'] + cv_df['loss_err'],
                     cv_df['loss_mean'] - cv_df['loss_err'],
                     color='C2', alpha=0.25)

ax_loss.plot(range(1, epochs+1), cv_df['val_loss_mean'], marker='None', color='C3',
             label='Training')
ax_loss.fill_between(range(1, epochs+1),
                     cv_df['val_loss_mean'] + cv_df['val_loss_err'],
                     cv_df['val_loss_mean'] - cv_df['val_loss_err'],
                     color='C3', alpha=0.25)

ax.set_xlabel('Training epochs')
ax.set_ylabel('Accuracy')
ax_loss.set_ylabel('Loss')

ax.grid()
ax.legend(title='Accuracy')
ax_loss.legend(title='Loss')

outfile = os.path.join(comp.paths.figures_dir, 'feature_engineering',
                       'accuracy-loss.png')
comp.check_output_dir(outfile)
plt.savefig(outfile)
plt.show()
#
#
# # In[24]:
#
#
# model = get_model(verbose=True)
# history = model.fit(X_train, y_train_cat,
#                     epochs=40,
#                     batch_size=batch_size,
#                     verbose=1)
#
#
# # In[31]:
#
#
# acc = history.history['acc']
# loss = history.history['loss']
#
#
# # In[32]:
#
#
# fig, ax = plt.subplots()
# ax.plot(range(1, 40+1), acc, marker='None', label='Training')
# # ax.plot(range(1, epochs+1), val_acc, marker='None', label='Validation')
# ax.grid()
# ax.legend()
# plt.show()
#
#
# # In[66]:
#
#
# pred_train = model.predict_proba(X_train)
# pred_val = model.predict_proba(X_val)
# pred_test = model.predict_proba(X_test)
#
#
# # In[67]:
#
#
# _four_group_encoding = {}
# _four_group_encoding['PPlus'] = 0
# _four_group_encoding['He4Nucleus'] = 1
# _four_group_encoding['O16Nucleus'] = 2
# _four_group_encoding['Fe56Nucleus'] = 3
#
#
# # In[68]:
#
#
# df_val_pred = pd.DataFrame(pred_val, columns=comp_list)
# df_val_pred['composition'] = [comp_list[i] for i in y_val]
# df_val_pred.head()
#
#
# # In[69]:
#
#
# df_test_pred = pd.DataFrame(pred_test, columns=comp_list)
# df_test_pred['composition'] = [comp_list[i] for i in y_test]
# df_test_pred.head()
#
#
# # In[70]:
#
#
# fig, ax = plt.subplots()
# for composition in comp_list:
#     label = _four_group_encoding[composition]
#     comp_mask = df_test_pred['composition'] == composition
#     a = np.log10(df_test_pred['PPlus']) - np.log10(df_test_pred['Fe56Nucleus'])
#     p_bins = np.linspace(-1, 1, 75)
#     ax.hist(a[comp_mask], bins=p_bins,
#                color=color_dict[composition], alpha=0.6,
#                label=composition)
# ax.set_xlabel('$\mathrm{p_{PPlus}/p_{Fe56Nucleus}}$')
# ax.set_ylabel('Counts')
# ax.grid()
# ax.legend()
# plt.show()
#
#
# # In[87]:
#
#
# fig, ax = plt.subplots()
# for composition in comp_list:
#     label = _four_group_encoding[composition]
#     comp_mask = df_sim_test.loc[~isnull_mask_test, f'comp_group_{num_groups}'] == composition
#     a = np.log10(df_test_pred['PPlus']) - np.log10(df_test_pred['Fe56Nucleus'])
#     p_bins = np.linspace(-1, 1, 75)
#     ax.scatter(a[comp_mask.values], df_sim_test.loc[~isnull_mask_test & comp_mask, 'log_s125'].values,
#                c=color_dict[composition], alpha=0.6,
#                label=composition)
# ax.set_xlabel('$\mathrm{p_{PPlus}/p_{Fe56Nucleus}}$')
# ax.set_ylabel('Counts')
# ax.grid()
# ax.legend()
# plt.show()
#
#
# # In[71]:
#
#
# g = sns.pairplot(df_test_pred, hue='composition', hue_order=comp_list,
#                  palette=[color_dict[c] for c in comp_list],
#                  markers='o',
#                  plot_kws=dict(alpha=0.1, lw=0),
#                  diag_kws=dict(bins=20))
# for i, j in zip(*np.triu_indices_from(g.axes, 1)):
#     g.axes[i, j].set_visible(False)
#
#
# # In[225]:
#
#
# fig, ax = plt.subplots()
# for composition in comp_list:
#     if composition in ['He4Nucleus', 'O16Nucleus']:
#         continue
#     label = _four_group_encoding[composition]
#     comp_mask = y_train == label
#     ax.scatter(pred_train[comp_mask, 0], pred_train[comp_mask, 3],
#                c=color_dict[composition], alpha=0.1,
#                label=composition)
# ax.set_xlabel('PPlus')
# ax.set_ylabel('Fe56Nucleus')
# ax.grid()
# ax.legend()
# plt.show()
#
#
# # In[202]:
#
#
# y_test
#
#
# # In[51]:
#
#
# score = model.evaluate(X_test, y_test_cat, batch_size=18)
# score
