!mkdir ia-rna
!wget -c 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
!mv pima-indians-diabetes.data.csv /content/ia-rna


# =============================================================================

import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

#Load Data
dataset = loadtxt('/content/ia-rna/pima-indians-diabetes.data.csv', delimiter=',')
#print (dataset)
#print (dataset.shape)

X = dataset[:,0:8]
Y  = dataset[:,8]
#print(X.shape, Y.shape)
#print(X)
#print(Y)


#Modelo
model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], activation='relu' ))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compilação do Modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit
model.fit(X , Y , epochs=150, batch_size=10)

#Verificar Aprendizado
#model.evaluate(X,Y)
_, accuracy = model.evaluate(X, Y)
print('ACCURACY: %.2f' % (accuracy * 100))
