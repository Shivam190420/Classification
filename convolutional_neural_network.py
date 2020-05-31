# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense
'''

# Part 1 - Data Preprocessing

# Generating images for the Training set
'''pixel take value B/E 0 and 255'''
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

'''rescale=1./255 will adjust all pixel value B/W 0 And 1'''

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('cat-and-dog/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('cat-and-dog/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

''' target_size = (64, 64) is size of image expected in CNN Model so below we 
chose 64,64 so here we will take same
batch_size is size of batches in which some random sample of our images will be
included so contain no of image which will go through CNN after which weights
will be updated
class_mode= bimary as 2 if more than 2 use catagorical'''

# Part 2 - Building the CNN
 
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))
''' no of filter = no of feature map
kernal_size=3 feature detector of 3X3 matrix
input_shape[64,64,3] for tensorflow backend 
input_shape[64,64,3] for theano backend
[dimension of 2D array=64,64  then no of channel=3]'''

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
''' stride=2 mean matrix will be slided by 2 columns
pool_size = 2 mean 2X2 matrix'''



#SECOND LATER---------------
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))




# Adding a third convolutional layer
'''DONE TO REDUCE SIZE MORE'''
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# 128 TAKEN AT RANDOM AS WE HAVE LOT OF DATA it should not be too small or too large


# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
'''optmizer = is algo we want to use to find optimial set of weights for NN
SCOCHASTIC GRADIENT DECENT ALGO adam is one of its part

loss = it is loss function within SGD [it is based on loss function which need
to be optmized for better result] (ex:- seen in Simple Linear Regression
       SUM(y-y')^2 -> min) ------> binary_cross.... for (0,1)   for more than 2
       used is catagorical_cross....
metrices = creterian we choose to evaluate our model
[so when weights are selected after each step then this creterian is 
used to improve model performance]     LIST IS CREATED BY USING [] BRACKET'''


# Training the CNN on the Training set and evaluating it on the Test set
j=cnn.fit_generator(training_set,
                  steps_per_epoch = 251,     # no of images in training set
                  epochs = 25,
                  validation_data = test_set, # correspond to test set on which we want to see performance of CNN
                  validation_steps = 64) # no of images in test set
# for graph
def AccuracyGraph():
    history=j
    epochs=25
    from matplotlib import pyplot as plt
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# FOR PREDECTION
def predection(d,p):
    import numpy as np
    from keras.preprocessing import image
    try:
        p=((d+'/')+p)
        test_image = image.load_img(p, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'Dog'
        else:
            prediction = 'Cat'
        return(prediction)
    except FileNotFoundError as m:
        return(m)
    
    
