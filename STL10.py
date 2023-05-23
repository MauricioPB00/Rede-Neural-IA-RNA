from google.colab import drive
drive.mount('/content/drive')
==============================
myPath = '/content/drive/'
==============================
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.layers import Dropout, BatchNormalization
==============================
def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        # Lê todos os bytes do arquivo
        data = f.read()
        
    # Converte os bytes para um array numpy
    labels = np.frombuffer(data, dtype=np.uint8)
    
    return labels
==============================
def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        # Lê todos os bytes do arquivo
        data = f.read()
        
    # Converte os bytes para um array numpy
    images = np.frombuffer(data, dtype=np.uint8)
    
    # Reshape e reordena as dimensões das imagens
    images = images.reshape((-1, 3, 96, 96)).transpose((0, 2, 3, 1))
    
    return images
==============================
HEIGHT = 96
WIDTH = 96
DEPTH = 3

TRAIN_DATA_PATH = 'train_X.bin'
TRAIN_LABEL_PATH = 'train_y.bin'
TEST_DATA_PATH = 'test_X.bin'
TEST_LABEL_PATH = 'test_y.bin'

train_data = read_all_images(myPath + TRAIN_DATA_PATH)
train_labels = read_labels(myPath + TRAIN_LABEL_PATH)

test_data = read_all_images(myPath + TEST_DATA_PATH)
test_labels = read_labels(myPath + TEST_LABEL_PATH)
==============================
print('formato do dataset de treino: ' + str(train_data.shape) + ', formato dos labels: ' + str(train_labels.shape))
print('formato do dataset de teste: ' + str(test_data.shape) + ', formato dos labels: ' + str(test_labels.shape))
==============================

def preparaDataset():
    # Preparação do dataset
    trainX = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 3)
    testX = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 3)

    trainy = train_labels - 1
    testy = test_labels - 1

    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    trainX = preProcessing(trainX)
    testX = preProcessing(testX)

    return trainX, testX, trainy, testy
==============================
def preProcessing(X):
  return (X.astype('float32') / 255)
==============================
def modelTraining(trainX, trainy, testX, testy):
    #model = defineModel()
    model = define_other_model()

    # Define os parâmetros do treinamento
    epochs = 80
    batch_size = 16
    steps_per_epoch = len(trainX)//batch_size

    datagen = ImageDataGenerator(rotation_range=20,
                             zoom_range=0.15,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.15,
                             horizontal_flip=True,
                             fill_mode="nearest")
    datagen.fit(trainX)
    history = model.fit(datagen.flow(trainX, trainy, batch_size=32),
                      epochs=epochs,
                      validation_data=(testX, testy),
                      steps_per_epoch=len(trainX) // 32,
                      batch_size=128).history

    # Treina o modelo
    #history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(testX, testy), verbose=1)
    #history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(testX, testy), steps_per_epoch=steps_per_epoch, verbose=1).history

    # Plota os gráficos de perda e precisão
    historyPrint(history)

    # Avalia o desempenho do modelo nos dados de teste
    modelEvaluation(model, testX, testy)

    return model, history
  #def train_model(X, y, model, epochs):
  #batch_size = 32
 
  ##return history
==============================
def define_other_model():
    reg=None
    ac='relu'
    num_filters=16
    drop_conv=0.5
    drop_dense=0.7

    # define VGG model
    model = Sequential()

    model.add(Conv2D(num_filters, (3, 3), activation=ac, kernel_regularizer=reg, input_shape=(96,96,3), padding='same'))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 16x16x3xnum_filters
    model.add(Dropout(drop_conv))
    

    model.add(Conv2D(2*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(2*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 8x8x3x(2num_filters)
    model.add(Dropout(drop_conv))

    model.add(Conv2D(4*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(4*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4num_filters)
    model.add(Dropout(drop_conv))

    model.add(Flatten())

    model.add(Dense(512, activation=ac,kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Dropout(drop_dense))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    return model
==============================

def defineModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(96, 96, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    return model
==============================
def modelEvaluation(model, testX, testy):
    _, accuracy = model.evaluate(testX, testy, verbose=0)
    print(f'Accuracy: {accuracy * 100:.3f}')
==============================
def historyPrint(history_dict):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plot loss
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()  # Clear the figure

    # Plot accuracy
    plt.plot(epochs, acc, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
==============================
def main():
    # Preparar o conjunto de dados
    trainX, testX, trainy, testy = preparaDataset()
    
    # Imprimir informações sobre o conjunto de dados
    print("Train data shape:", trainX.shape)
    print("Train labels shape:", trainy.shape)
    print("Test data shape:", testX.shape)
    print("Test labels shape:", testy.shape)
    
    # Treinar o modelo
    modelTraining(trainX, trainy, testX, testy)

==============================
main()
==============================

==============================


