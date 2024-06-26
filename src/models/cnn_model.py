from keras import Input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.optimizers import Adam


def create_model(input_shape, num_classes):
    model = Sequential()

    model.add(Input(shape=input_shape))

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='valid', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten layer
    model.add(Flatten())

    # Dense layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model

# Define input shape and number of classes
input_shape = (400, 224, 1)  # Assuming input images are grayscale
num_classes = 24  # Change this according to the number of classes in your dataset

# Create the model
model = create_model(input_shape, num_classes)

# Print model summary
# model.summary()


# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#
# def create_model(input_shape, num_classes):
#     model = Sequential()
#
#     # Convolutional layers
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     # Flatten layer
#     model.add(Flatten())
#
#     # Dense layer
#     model.add(Dense(64, activation='relu'))
#
#     # Output layer
#     model.add(Dense(num_classes, activation='softmax'))
#
#     # Compile the model
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     return model
#
# # Define input shape and number of classes
# input_shape = (224, 224, 1)
# num_classes = 23  # Change this according to the number of classes in your dataset
#
# # Create the smaller model
# model = create_model(input_shape, num_classes)
#
# # Print model summary
# model.summary()
#