# -------------------------------------------------------#
# Template for creating and testing different models of  #
# neural networks using the same dataset.                #
# When training models, from training dataset, script    #
# allocates 10% to validate the outputs.                 #
# After training, the script tests the performance       #
# of the model on a test dataset (separate data file)    #
# -------------------------------------------------------#
# Import Libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import MetaTrader5 as mt5

# Add fonts
font_list=fm.findSystemFonts()
for f in font_list:
    if(f.__contains__('ClearSans')):
        fm.fontManager.addfont(f)

# Set parameters for output graphs
mp.rcParams.update({'font.family':'serif',
                    'font.serif':'Clear Sans',
                    'axes.titlesize': 'x-large',
                    'axes.labelsize':'medium',
                    'xtick.labelsize':'small',
                    'ytick.labelsize':'small',
                    'legend.fontsize':'small',
                    'figure.figsize':[6.0,4.0],
                    'axes.titlecolor': '#707070',
                    'axes.labelcolor': '#707070',
                    'axes.edgecolor': '#707070',
                    'xtick.labelcolor': '#707070',
                    'ytick.labelcolor': '#707070',
                    'xtick.color': '#707070',
                    'ytick.color': '#707070',
                    'text.color': '#707070',
                    'lines.linewidth': 0.8,
                    'axes.linewidth': 0.5
                   })

# connect to the MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# request the path to the sandbox
path=os.path.join(mt5.terminal_info().data_path,r'MQL5\Files')
mt5.shutdown()

# Load training dataset
filename = os.path.join(path,'study_data.csv')
data = np.asarray( pd.read_table(filename,
                   sep=',',
                   header=None,
                   skipinitialspace=True,
                   encoding='utf-8',
                   float_precision='high',
                   dtype=np.float64,
                   low_memory=False))

# Split training dataset to input data and target
inputs=data.shape[1]-2
targerts=2
train_data=data[:,0:inputs]
train_target=data[:,inputs:]

print(data)

# create a neural network model
model = keras.Sequential([keras.layers.InputLayer(input_shape=inputs),
                         # Fill the model with a description of the neural layers
                          ])
model.compile(optimizer='Adam', 
               loss='mean_squared_error', 
               metrics=['accuracy'])
               
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
history = model.fit(train_data, train_target,
                      epochs=500, batch_size=1000,
                      callbacks=[callback],
                      verbose=2,
                      validation_split=0.1,
                      shuffle=True)
                      
# save the trained model
model.save(os.path.join(path,'model.h5'))

# plot model training results
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.ylabel('$MSE$ $Loss$')
plt.xlabel('$Epochs$')
plt.title('Model training dynamics')
plt.legend(loc='upper right')

plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.ylabel('$Accuracy$')
plt.xlabel('$Epochs$')
plt.title('Model training dynamics')
plt.legend(loc='lower right')

# Load testing dataset
test_filename = os.path.join(path,'test_data.csv')
test = np.asarray( pd.read_table(test_filename,
                   sep=',',
                   header=None,
                   skipinitialspace=True,
                   encoding='utf-8',
                   float_precision='high',
                   dtype=np.float64,
                   low_memory=False))

# Split test dataset to input data and target
test_data=test[:,0:inputs]
test_target=test[:,inputs:]

# check model results on a test dataset
test_loss, test_acc = model.evaluate(test_data, test_target) 

# Log testing results
print('Model in test')
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# output of graphs
plt.show()