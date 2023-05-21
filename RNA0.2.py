
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

#Load Data
dataset = loadtxt('/content/ia-rna/pima-indians-diabetes.data.csv', delimiter=',')
#print (dataset)
#print (dataset.shape)

X = dataset[:,0:8]
y  = dataset[:,8]
#print(X.shape, Y.shape)
#print(X)
#print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80, shuffle=True)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#Modelo
model = Sequential()
model.add(Dense(20, input_dim=X.shape[1], activation='relu' ))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compilação do Modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit
history = model.fit(X_train, y_train, validation_data=(X_test, y_test) , epochs=150, batch_size=50)
print(history.history.keys())

#accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left' )
plt.show()

#Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left' )
plt.show()


#Verificar Aprendizado

#_, accuracy = model.evaluate(X_test, y_test)
#model.evaluate(X,Y)
#print('ACCURACY: %.2f' % (accuracy * 100))

pred = model.predict(X_test)
print(pred)

#print([y_test == pred])
cm = confusion_matrix(y_test, pred)
print(cm)

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

acc = (tp + tn)  / (tp + tn + fn + fp)

print("TPR" , tpr)
print("TNR ", tnr)
print("ACC ", acc)

#AUC
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=0)
auc = metrics.auc(fpr, tpr)

#codigo para plotar o grafico da curva ROC
plt.plot([0,1], [0,1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.plot(fpr,tpr,color='b', label=r'ROC {AUC =%0.2f}' % (auc), lw=2, alpha=.8)
plt.suptitle("ROC Curve")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.show()
