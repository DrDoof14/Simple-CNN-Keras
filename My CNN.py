from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
get_ipython().magic(u'config IPCompleter.greedy=True')


#part 1 : building the CNN
#training
classifier=Sequential()



classifier.add(Conv2D(filters=32,kernel_size=(3,3),data_format='channels_last',input_shape=(64,64,3),activation='relu'))



#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#adding a second conv layer
classifier.add(Conv2D(filters=32,kernel_size=(3,3),data_format='channels_last',input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#flattening
classifier.add(Flatten())



#fully connected layers
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))


#compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#part 2 :fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)



