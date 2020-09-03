#!/usr/bin/env python

from __future__ import division
from squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np
import os

training_dir = 'data/train'
val_dir = 'data/validation'
weights_file = 'squeezenet_weights.h5'

initial_epoch = 0
nb_epoch = 50
batch_size = 128
samples_per_epoch = 1920
nb_val_samples = 100

input_shape = (224, 224)
channels = 3

#fix class order (0=RED, 1=YELLOW, 2=GREEN, 4=UNKOWN)
classes = ['0', '1', '2', '4']

nb_classes = len(classes)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        width_shift_range=0.1,
        zoom_range=0.2,
        height_shift_range=0.1,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(training_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    classes=classes)
print('train datagen class indices: \n%s' % train_generator.class_indices)

val_generator = test_datagen.flow_from_directory(val_dir,
                                                 target_size=input_shape,
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 classes=classes)
print('val datagen class indices: \n%s' % val_generator.class_indices)

print('Loading model..')
model = SqueezeNet(nb_classes, input_shape[0], input_shape[1], channels)
model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy', 'categorical_crossentropy'])
if os.path.isfile(weights_file):
    print('Loading weights: %s' % weights_file)
    model.load_weights(weights_file, by_name=True)
else:
    # if it is a new model, set balanced bias for the last layer
    w, _ = model.get_layer('conv10').get_weights()
    model.get_layer('conv10').set_weights([w,np.log(class_weight/sum(class_weight))])

print('Fitting model')
# Balance class weights by frequency and give preference for RED lights.
class_weight = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)
class_weight[0] *= 2

model.fit_generator(train_generator,
                    samples_per_epoch=samples_per_epoch,
                    validation_data=val_generator,
                    validation_steps=nb_val_samples//64,
                    nb_epoch=nb_epoch,
                    verbose=1,
                    initial_epoch=initial_epoch,
                    class_weight = class_weight)
print("Finished fitting model")

print('Saving weights')
model.save_weights(weights_file, overwrite=True)
print('Evaluating model')
score = model.evaluate_generator(val_generator, steps=int(samples_per_epoch/nb_val_samples))
print('result: %s' % score)

