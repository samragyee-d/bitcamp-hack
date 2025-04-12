from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model
model = load_model("emotion_model.h5")
base_path = "images/"

# number of images to feed into the NN for every batch
batch_size = 128
# size of the image: 48*48 pixels
pic_size = 48

datagen_validation = ImageDataGenerator()

# Create the validation generator
validation_generator = datagen_validation.flow_from_directory(base_path + "validation",
                                                    target_size=(pic_size,pic_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

# Evaluate the model on the validation data
loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.n // validation_generator.batch_size)

# Print the results
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
