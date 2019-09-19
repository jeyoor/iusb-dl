from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,CSVLogger


DATASET_PATH  = './dataset'
IMAGE_SIZE    = (224, 224)
#IMAGE_SIZE    = (32, 32)
NUM_CLASSES   = 2
BATCH_SIZE    = 32  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 100
WEIGHTS_FINAL = 'model-dogcat-functional-final.h5'


train_datagen = ImageDataGenerator( rescale=1.0 / 255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                    channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

#Now make our first convolutional layer, 32 filters, 3x3, default stride and padding
inputs = Input(shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],3))
layer1 = Conv2D(32, (3, 3), activation='relu')(inputs)
#Do a max pooling, increase number of filters, but reduce feature map size
layer2 = MaxPooling2D(pool_size=(2,2))(layer1)
#And another convolution/pooling pair
layer3 = Conv2D(64, (3, 3), activation='relu')(layer2)
layer4 = MaxPooling2D(pool_size=(2,2))(layer3)
#Finally another convolutional layer
layer5 = Conv2D(128, (3,3), activation='relu')(layer4)
#Make output of prior convolutional layer 1D for Dense layers
layer6 = Flatten()(layer5)
#Dense layers form the classifier using input from convolutional layers
layer7 = Dense(32, activation='sigmoid')(layer6) #Could be 'relu' activation
layer8 = Dense(2, activation='softmax')(layer7) #Softmax represents outputs as probabilites

#Use Adam optimizer (instead of plain SGD), set learning rate to explore.
adam = Adam(lr=.001)

#instantiate model
model = Model(inputs=inputs, outputs=layer8)
#Compile model
model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])

#Print layers for resulting model
model.summary()

#Log training data into csv file
csv_logger = CSVLogger(filename="log.csv")
cblist = [csv_logger]

# train the model
model.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
			callbacks=cblist)

# save trained model and weights
model.save(WEIGHTS_FINAL)

#Generators adapted from:
#MIT License
#
#Copyright (c) 2018 JK Jung
#Copyright (c) 2019 James Wolfer
#Copyright (c) 2019 Jeyan Burns-Oorjitham
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
