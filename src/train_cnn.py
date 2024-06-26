import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.models.cnn_model import model
from src.preprocessing.image_processor import ImageProcessor

# Define directories for training and validation data
# dataset_dir = 'C:\\Users\\bellv\\EchoFlowDataset\\asl\\fingerspelling5\\dataset5\\A'
dataset_dir = 'C:\\Users\\bellv\\EchoFlowDataset\\DatasetPropio\\Ezi AI'
validation_dir = dataset_dir
additional_validation_dir = dataset_dir
# validation_dir = 'C:\\Users\\bellv\\EchoFlowDataset\\asl\\fingerspelling5\\dataset5\\B'
# additional_validation_dir = 'C:\\Users\\bellv\\EchoFlowDataset\\asl\\fingerspelling5\\dataset5\\C'

# Define image dimensions and batch size
img_width, img_height = 400, 224
batch_size = 7

# Instantiate ImageProcessor
image_processor = ImageProcessor()

# Add preprocessing steps
image_processor.add_processing_step(lambda img: ImageProcessor.denoise_image(img, kernel_size=(21, 21)))
image_processor.add_processing_step(lambda img: ImageProcessor.resize_image(img, (img_width, img_height)))
image_processor.add_processing_step(ImageProcessor.normalize_image)
image_processor.add_processing_step(ImageProcessor.edge_detection)
image_processor.add_processing_step(ImageProcessor.contour_detection)
image_processor.add_processing_step(ImageProcessor.normalize_image)
image_processor.add_processing_step(image_processor.extract_sift_features)
image_processor.add_processing_step(image_processor.suppress_non_roi)
image_processor.add_processing_step(ImageProcessor.normalize_image)

# Use ImageDataGenerator to preprocess images
data_gen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=image_processor.process_image  # Note the function call here
)

# Generate batches of data for training
train_generator = data_gen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# Generate batches of data for validation
validation_generator = data_gen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# Generate batches of data for additional validation
additional_validation_generator = data_gen.flow_from_directory(
    additional_validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# Convert to tf.data.Dataset
def generator(data_gen):
    for batch in data_gen:
        yield batch[0], batch[1]

# Create datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: generator(train_generator),
    output_signature=(
        tf.TensorSpec(shape=(None, img_width, img_height, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32)
    )
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: generator(validation_generator),
    output_signature=(
        tf.TensorSpec(shape=(None, img_width, img_height, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, validation_generator.num_classes), dtype=tf.float32)
    )
).repeat()

additional_validation_dataset = tf.data.Dataset.from_generator(
    lambda: generator(additional_validation_generator),
    output_signature=(
        tf.TensorSpec(shape=(None, img_width, img_height, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, additional_validation_generator.num_classes), dtype=tf.float32)
    )
).repeat()

# Define steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size
additional_validation_steps = additional_validation_generator.samples // batch_size

# Train the model
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=validation_steps,
    epochs=30
)

# Validate the model with the additional validation dataset
additional_validation_results = model.evaluate(
    additional_validation_dataset,
    steps=additional_validation_steps
)

print(f"Additional validation results: {additional_validation_results}")

# Save the trained model
model.save('bsl_fingerspelling_cnn_model.keras')

# Plot training accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Function to predict and log video frames (commented out)
# def predict_video(video_path, model, image_processor, output_log='predictions.log'):
#     cap = cv2.VideoCapture(video_path)
#     frame_rate = cap.get(cv2.CAP_PROP_FPS)
#     predictions = []
#     prev_prediction = None
#     prev_time = 0

#     with open(output_log, 'w') as log_file:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Get the timestamp of the current frame
#             timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds

#             # Process the frame
#             processed_frame = image_processor.process_image(frame)
#             processed_frame = np.expand_dims(processed_frame, axis=0)

#             # Make predictions
#             preds = model.predict(processed_frame)
#             predicted_class = np.argmax(preds)

#             # Log the prediction if it changes
#             if predicted_class != prev_prediction:
#                 log_file.write(f"[{predicted_class}, {str(datetime.timedelta(seconds=timestamp))}]\n")
#                 prev_prediction = predicted_class

#     cap.release()

# video_path = "C:\\Users\\bellv\\EchoFlowDataset\\fingerspelling\\Finger-spelling.mp4"
# predict_video(video_path, model, image_processor, output_log='predictions.log')
