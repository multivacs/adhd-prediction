"""
ADHD PREDICTION WITH NEURAL NETWORKS

This script is designed to configure with your own data and specify the different analysis you want to make.
For example, if you don't want to make an Hyperopt analysis, you can disable it by setting the flag to False. 

Author: Mario
Last Edited: 2024-02-26
"""

import math
import time
import datetime
import pandas as pd
import numpy as np
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import sys
script_dir = os.path.dirname(__file__)
rel_path = f'models'
path = os.path.join(script_dir, rel_path)
sys.path.append(path)

# Models
from models.tabnet import TabNet
from models.tabtransformer import TabTransformer
from models.node import Node
from models.DCNN import DCNN


#-----------------------------------------------------------------------------------------------
# Global Configuration Variables
# These variables define the parameters and workflows applied for the rest of the script.
#-----------------------------------------------------------------------------------------------
FEATURE_ENGINEERING = False     # Do or not simple data analysis of the data (comparision between IQs of clinic and control groups, correlation matrix)
HYPEROPT = False                # Do or not hyperopt adjustment
EVALUATE = True                 # Do or not evaluate the models with KFold
SCALER_NORMALIZER = "Standard"  # Normalization of numeric features Standard/Robust
NUM_FOLDS = 10                  # Number of folds applied in KFold
NUM_EPOCHS = 100                # Number of epochs done for each kfold training
HO_EPOCHS = 100                 # Hyperopt epochs
HO_MAXEVALS = 60                # Hyperopt max evaluations
HO_LOSS_IMPROVE = 10            # Hyperopt early stop epochs without loss improvement
SEED = 1234                     # Random seed


#-----------------------------------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# These are the hyperparameters founded that works best with my configuration and data.
#-----------------------------------------------------------------------------------------------
best_tabnet = {'bn_momentum': 0.942772441153329, 'feature_dim': 128, 'learning_rate': 0.03737086735655166, 'n_steps': 5, 'relaxation_factor': 1.2000000000000002, 'sparsity': 0.0595285322427291}
best_tabtransformer = {'depth': 4, 'dropout': 0.28774402657808856, 'embedding_dim': 64, 'heads': 8, 'learning_rate': 0.0015283238179232778, 'weight_decay': 0.07992657295500534}
best_node = {'depth': 2, 'layer_dim': 384, 'learning_rate': 0.0012095595812000894, 'num_layers': 6}
best_1dcnn = {'K': 8, 'cha_hidden': 128, 'cha_input': 192, 'drop_hidden': 0.11412807906220623, 'drop_input': 0.250317695794937, 'drop_out': 0.23413921322423242, 'sign_size': 128}


#---------------------------------------------------------------------------------------------------
# DATASET SPECIFIC
# Here you define the features that apply to your dataset, and the target feature (label supervised).
#---------------------------------------------------------------------------------------------------
# Column information
NUMERIC_FEATURES = ['BD', 'SI', 'DS', 'PCn', 'CD', 'VC', 'LN', 'MR', 'CO', 'SS', 'VCI', 'PRI', 'WMI', 'PSI', 'FSIQ', 'GAI', 'CPI', 'GAI-CPI', 'WMI-PSI', 'GAI-WMI', 'GAI-FSIQ', 'PRI-CPI', 'FSIQ-CPI', 'VCI-CPI', 'PRI-WMI']
CATEGORICAL_FEATURES = []
FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
LABEL = 'ADHD'

# Read dataset
df = pd.read_csv('./dataset.csv', sep=';', decimal=',')

# Drop duplicated samples
df.drop_duplicates()


################ FEATURE ENGINEERING ###############

if FEATURE_ENGINEERING:
  
  # Comparison between clinic/control groups
  x = ['FSIQ', 'VCI', 'PRI', 'WMI', 'PSI', 'GAI', 'CPI']  # Features to analyze
  g_clinic = df.loc[df[LABEL] == 1]
  g_control = df.loc[df[LABEL] == 0]
  
  g_clinic = g_clinic[x].mean(axis=0).to_numpy()
  g_control = g_control[x].mean(axis=0).to_numpy()

  plt.plot(x, g_clinic, label='Grupo clínico')
  plt.plot(x, g_control, label='Grupo control')
  plt.legend()
  plt.show()

  # Correlation matrix
  cormat = df.corr()
  print(cormat)
  sns.heatmap(cormat, annot=True)


################ Data Preprocessing ###############

# Separate target label
X = df.drop([LABEL],axis =1)
y = df[LABEL].astype('int64')

# Standard Scaler
if SCALER_NORMALIZER == "Standard":
  scaler = StandardScaler()
  X.loc[:, NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])

# Robust Scaler
elif SCALER_NORMALIZER == "Robust":
  scaler = RobustScaler(quantile_range=(20, 80))
  X.loc[:, NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])


################ Hyperparameter Tuning ###############
  
if HYPEROPT == True:
  best_tabnet = {}
  best_tabtransformer = {}
  best_node = {}
  best_1dcnn = {}

  ho_time = time.time()

  # Define searching space
  TabNet()
  space_tabnet = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'feature_dim': hp.quniform('feature_dim', 64, 256, 64),
    'n_steps': hp.quniform('n_steps', 2, 6, 1),
    'relaxation_factor': hp.quniform('relaxation_factor', 1, 3, q=0.1),
    'sparsity': hp.uniform('sparsity', 0.001, 0.1),
    'bn_momentum': hp.uniform('bn_momentum', 0.9, 0.9999)
  }

  space_tabtransformer = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'weight_decay': hp.uniform('weight_decay', 0.0001, 0.1),
    'embedding_dim': hp.quniform('embedding_dim', 16, 64, 16),
    'depth': hp.quniform('depth', 2, 8, 1),
    'heads': hp.quniform('heads', 4, 16, 1),
    'dropout': hp.uniform('dropout', 0.1, 0.3)
  }

  space_node = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'layer_dim': hp.quniform('layer_dim', 128, 512, 128),
    'num_layers': hp.quniform('num_layers', 2, 6, 1),
    'depth': hp.quniform('depth', 2, 6, 2)
  }

  space_1dcnn = {
    'sign_size': hp.quniform('sign_size', 16, 128, 16),
    'cha_input': hp.quniform('cha_input', 64, 256, 64),
    'cha_hidden': hp.quniform('cha_hidden', 32, 128, 32),
    'K': hp.quniform('K', 2, 8, 1),
    'drop_input': hp.uniform('drop_input', 0.1, 0.3),
    'drop_hidden': hp.uniform('drop_hidden', 0.1, 0.3),
    'drop_out': hp.uniform('drop_out', 0.1, 0.3)
  }


  # Objective functions
  def objective_tabnet(space):
    model = TabNet(learning_rate=space['learning_rate'],
                   feature_dim=int(space['feature_dim']),
                   n_steps=int(space['n_steps']),
                   relaxation_factor=space['relaxation_factor'],
                   sparsity=space['sparsity'],
                   bn_momentum=space['bn_momentum'])
    
    roc = model.fit(X, y, epochs=HO_EPOCHS, random_state=SEED)

    # Negative loss for maximization problems
    return{'loss': -roc, "status": STATUS_OK}

  def objective_tabtransformer(space):
    model = TabTransformer(learning_rate=space['learning_rate'],
                           weight_decay=space['weight_decay'],
                           embedding_dim=int(space['embedding_dim']),
                           depth=int(space['depth']),
                           heads=int(space['heads']),
                           dropout=space['dropout'],
                           NUMERIC_FEATURES = NUMERIC_FEATURES, CATEGORICAL_FEATURES = CATEGORICAL_FEATURES, LABEL = LABEL)

    roc = model.fit(X, y, epochs=HO_EPOCHS, random_state=SEED)

    return{'loss': -roc, "status":STATUS_OK}

  def objective_node(space):
    model = Node(learning_rate=space['learning_rate'],
                 layer_dim=int(space['layer_dim']),
                 num_layers=int(space['num_layers']),
                 depth=int(space['depth']))

    roc = model.fit(X, y, epochs=HO_EPOCHS, random_state=SEED)

    return{'loss': -roc, "status":STATUS_OK}

  def objective_1dcnn(space):
    model = DCNN(sign_size=int(space['sign_size']),
                 cha_input=int(space['cha_input']),
                 cha_hidden=int(space['cha_hidden']),
                 K=int(space['K']),
                 drop_input=space['drop_input'],
                 drop_hidden=space['drop_hidden'],
                 drop_out=space['drop_out'])

    roc = model.fit(X, y, epochs=HO_EPOCHS, random_state=SEED)

    return{'loss': -roc, "status":STATUS_OK}


  # Searching
  trials = Trials()
  time_tabnet = time.time()
  best_tabnet = fmin( fn = objective_tabnet, space=space_tabnet, algo= tpe.suggest, max_evals=HO_MAXEVALS, trials=trials, early_stop_fn=no_progress_loss(HO_LOSS_IMPROVE))
  time_tabnet = str(datetime.timedelta(seconds=time.time() - time_tabnet))
  loss_tabnet = trials.best_trial['result']['loss']
  print(f"TABNET HYPEROPT = {best_tabnet}")

  trials = Trials()
  time_tabtransformer = time.time()
  best_tabtransformer = fmin( fn = objective_tabtransformer, space=space_tabtransformer, algo= tpe.suggest, max_evals=HO_MAXEVALS, trials=trials, early_stop_fn=no_progress_loss(HO_LOSS_IMPROVE))
  time_tabtransformer = str(datetime.timedelta(seconds=time.time() - time_tabtransformer))
  loss_tabtransformer = trials.best_trial['result']['loss']
  print(f"TABTRANSFORMER HYPEROPT = {best_tabtransformer}")

  trials = Trials()
  time_node = time.time()
  best_node = fmin( fn = objective_node, space=space_node, algo= tpe.suggest, max_evals=HO_MAXEVALS, trials=trials, early_stop_fn=no_progress_loss(HO_LOSS_IMPROVE))
  time_node = str(datetime.timedelta(seconds=time.time() - time_node))
  loss_node = trials.best_trial['result']['loss']
  print(f"NODE HYPEROPT = {best_node}")

  trials = Trials()
  time_1dcnn = time.time()
  best_1dcnn = fmin( fn = objective_1dcnn, space=space_1dcnn, algo= tpe.suggest, max_evals=HO_MAXEVALS, trials=trials, early_stop_fn=no_progress_loss(HO_LOSS_IMPROVE))
  time_1dcnn = str(datetime.timedelta(seconds=time.time() - time_1dcnn))
  loss_1dcnn = trials.best_trial['result']['loss']
  print(f"1DCNN HYPEROPT = {best_1dcnn}")


  # Save results
  ho_time = time.time() - ho_time
  ho_time = str(datetime.timedelta(seconds=ho_time))
  print(f'Execution time HyperOpt: {ho_time}')

  now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
  ts = math.floor(time.time())
  script_dir = os.path.dirname(__file__)
  rel_path = f'hyperopt\\HO_{ts}.txt'
  filepath = os.path.join(script_dir, rel_path)

  f = open(filepath, "w")
  output = f'# {now}\n# Hyperparameter tuning with: [Exec. time={ho_time}, Epochs={HO_EPOCHS}, Max evals={HO_MAXEVALS}]\n\n'
  output += f'TABNET HYPEROPT = {best_tabnet}\ntime={time_tabnet} ; best_loss={loss_tabnet}\n\n'
  output += f'TABTRANSFORMER HYPEROPT = {best_tabtransformer}\ntime={time_tabtransformer} ; best_loss={loss_tabtransformer}\n\n'
  output += f'NODE HYPEROPT = {best_node}\ntime={time_node} ; best_loss={loss_node}\n\n'
  output += f'1DCNN HYPEROPT = {best_1dcnn}\ntime={time_1dcnn} ; best_loss={loss_1dcnn}'
  f.write( output )
  f.close()


################ KFOLD ###############
  
if EVALUATE == False:
  sys.exit()

# Create models
names = ["TabNet", "TabTransformer", "Node", "1DCNN"]
models = [TabNet(learning_rate=best_tabnet['learning_rate'],
                 feature_dim=int(best_tabnet['feature_dim']),
                 n_steps=int(best_tabnet['n_steps']),
                 relaxation_factor=best_tabnet['relaxation_factor'],
                 sparsity=best_tabnet['sparsity'],
                 bn_momentum=best_tabnet['bn_momentum']),

          TabTransformer(learning_rate=best_tabtransformer['learning_rate'],
                         weight_decay=best_tabtransformer['weight_decay'],
                         embedding_dim=int(best_tabtransformer['embedding_dim']),
                         depth=int(best_tabtransformer['depth']),
                         heads=int(best_tabtransformer['heads']),
                         dropout=best_tabtransformer['dropout'],
                         NUMERIC_FEATURES = NUMERIC_FEATURES, CATEGORICAL_FEATURES = CATEGORICAL_FEATURES, LABEL = LABEL),

          Node(learning_rate=best_node['learning_rate'],
               layer_dim=int(best_node['layer_dim']),
               num_layers=int(best_node['num_layers']),
               depth=int(best_node['depth'])),

          DCNN(sign_size=int(best_1dcnn['sign_size']),
               cha_input=int(best_1dcnn['cha_input']),
               cha_hidden=int(best_1dcnn['cha_hidden']),
               K=int(best_1dcnn['K']),
               drop_input=best_1dcnn['drop_input'],
               drop_hidden=best_1dcnn['drop_hidden'],
               drop_out=best_1dcnn['drop_out'])
          ]


all_results = []
for model, name in zip(models, names):
  t_start = time.time()
  result, confusion = model.fit_evaluate(X, y, cv=NUM_FOLDS, epochs=NUM_EPOCHS, random_state = SEED)
  t_end = time.time() - t_start
  thisdict = dict(model_name = name, score = result, confmatrix = confusion, avg_score = np.average(result, axis=0), execution_time_sec = t_end)
  all_results.append(thisdict)


# Print results
df_result = pd.DataFrame.from_dict(all_results)
df_result['accuracy'] = list(zip(*df_result['avg_score'].values))[0]
df_result['auc_roc'] = list(zip(*df_result['avg_score'].values))[1]
df_result['recall'] = list(zip(*df_result['avg_score'].values))[2]
df_result['specificity'] = list(zip(*df_result['avg_score'].values))[3]
df_result = df_result.drop(columns=['score'])
df_result = df_result.drop(columns=['avg_score'])
df_result = df_result.drop(columns=['confmatrix'])


print(df_result)


# Save results
now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
ts = math.floor(time.time())
script_dir = os.path.dirname(__file__)
rel_path = f'results\\{ts}.txt'
filepath = os.path.join(script_dir, rel_path)

f = open(filepath, "w")
f.write( f'# {now}\n# Results for: [Hyperparameters: True, Preprocess_scaler: {SCALER_NORMALIZER}, Folds: {NUM_FOLDS}, Epochs: {NUM_EPOCHS}, Seed: {SEED}]\n\n' +
        str(all_results[0]) + '\n' + str(all_results[1]) + '\n' + str(all_results[2]) + '\n' + str(all_results[3]))
f.close()