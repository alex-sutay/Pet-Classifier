import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from sklearn import model_selection

import os
from PIL import Image

DATA_DIR = 'D:/Downloads/Pets Image Data'
SIZE = 200


# Load the data
if os.path.exists('x.npy') and os.path.exists('y.npy'):
    x = np.load('x.npy')
    y = np.load('y.npy')
else:
    x = None
    y = None
    classes = os.listdir(DATA_DIR)
    num_cls = len(classes)
    if 'other' in classes:
        num_cls -= 1
    this_cls = -1
    for clsdir in classes:
        print(clsdir)
        if clsdir != 'other':
            this_cls += 1
        for filename in os.listdir(DATA_DIR + "/" + clsdir):
            print(filename, end='\r')
            if filename.endswith((".jpg", ".png", ".jpeg")):
                # Create the new X
                im = Image.open(DATA_DIR + "/" + clsdir + "/" + filename, 'r')
                image = im.resize((SIZE, SIZE))
                l_im = image.convert('L')
                this_array = np.asarray(l_im)
                im.close()
                this_array.resize(1, SIZE, SIZE)
                # Add the new X
                if x is None:
                    x = this_array
                else:
                    x = np.vstack([x, this_array])
                # create the corresponding Y
                this_y = np.zeros(num_cls)
                if clsdir != 'other':
                    this_y[this_cls] = 1
                # Add the new Y
                if y is None:
                    y = this_y
                else:
                    y = np.vstack([y, this_y])
    np.save('x.npy', x)
    np.save('y.npy', y)

print(x)
print(x.shape)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=(SIZE, SIZE, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics = ['Accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1, validation_data=(x_test, y_test))
model.save('new_model')

