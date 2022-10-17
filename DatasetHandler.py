import ImageHandler as ih
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
# Build the arrays for both datasets and then datagen.flow from arrays
def build_dataset_arrays(validation_split, list_of_image_paths, absolute_limit, target_shape):
    train_val_limit = int(absolute_limit * (1 - validation_split))

    train_label_array = []
    validation_label_array = []
    train_input_array = []
    validation_input_array = []

    counter = 0
    for index, image_path in enumerate(list_of_image_paths):
        label_image_tensor = ih.load_image_label(image_path, target_shape)
        if not label_image_tensor.shape == (256, 256, 3):
            continue
        input_image_tensor = ih.add_synthetic_noise(label_image_tensor)
        if counter < train_val_limit:
            train_label_array.append(label_image_tensor)
            train_input_array.append(input_image_tensor)
        else:
            validation_label_array.append(label_image_tensor)
            validation_input_array.append(input_image_tensor)
        counter += 1
        if counter%200 == 0:
            print("Counter is at {}".format(counter))
            pass
        if index >= absolute_limit-1:
            break
        pass
    #Convert them into numpy arrays
    train_label_array = np.array(train_label_array)
    validation_label_array = np.array(validation_label_array)
    train_input_array = np.array(train_input_array)
    validation_input_array = np.array(validation_input_array)

    return train_input_array, train_label_array, validation_input_array, validation_label_array

def build_ImageDataGenerators(list_of_image_paths, validation_split, absolute_limit, target_shape, batch_size):
    train_input_array, train_label_array, validation_input_array, validation_label_array = build_dataset_arrays(validation_split=validation_split, list_of_image_paths=list_of_image_paths, absolute_limit=absolute_limit, target_shape=target_shape)

    train_datagen = ImageDataGenerator()
    validation_datagen = ImageDataGenerator()
    train_data_generator = train_datagen.flow(x=train_input_array, y=train_label_array, batch_size=batch_size, shuffle=True)
    validation_data_generator = validation_datagen.flow(x=validation_input_array, y=validation_label_array, batch_size=batch_size)
    return train_data_generator, validation_data_generator