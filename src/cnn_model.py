from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def create_model(input_shape, num_classes):
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))

    # Second convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu'))

    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout layer
    model.add(Dropout(0.25))

    # Third convolutional layer
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))

    # Fourth convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout layer
    model.add(Dropout(0.25))

    # Fifth convolutional layer
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))

    # Sixth convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu'))

    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout layer
    model.add(Dropout(0.25))

    # Seventh convolutional layer
    model.add(Conv2D(256, (3, 3), padding='valid', activation='relu'))

    # Eighth convolutional layer
    model.add(Conv2D(256, (3, 3), activation='relu'))

    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout layer
    model.add(Dropout(0.25))

    # Ninth convolutional layer
    model.add(Conv2D(512, (3, 3), padding='valid', activation='relu'))

    # Tenth convolutional layer
    model.add(Conv2D(512, (3, 3), activation='relu'))

    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout layer
    model.add(Dropout(0.25))

    # Flatten layer
    model.add(Flatten())

    # Dense layer
    model.add(Dense(512, activation='relu'))

    # Dropout layer
    model.add(Dropout(0.25))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model

# Define input shape and number of classes
input_shape = (224, 224, 1)  # Assuming input images are RGB
num_classes = 10  # Change this according to the number of classes in your dataset

# Create the model
model = create_model(input_shape, num_classes)

# Print model summary
model.summary()
