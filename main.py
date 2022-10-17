import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import DatasetHandler as dh
import ImageHandler as ih
import Autoencoder_model as autoencoder


folder_path = "C:\\Users\\allan\\OneDrive\\Desktop\\Landscape_Images"
list_of_image_names = os.listdir(folder_path)
list_of_image_paths = [os.path.join(folder_path, name) for name in list_of_image_names]
del list_of_image_names

#Set up the datasets
validation_split = 0.1
train_data_generator, validation_data_generator = dh.build_ImageDataGenerators(list_of_image_paths, validation_split, absolute_limit=3500, target_shape=(256, 256), batch_size=16)

#Set up the model
my_autoencoder = autoencoder.Autoencoder_model((256, 256, 3), train_data_generator, validation_data_generator)
my_autoencoder.build_whole_model()
my_autoencoder.show_model_summary()
my_autoencoder.compile_model()
#Train the model
my_autoencoder.train_model(epochs=20)

#Set up a test image
test_image_path = list_of_image_paths[6]
test_image_tensor = ih.load_image_label(test_image_path, (256, 256))
test_image_tensor = ih.add_synthetic_noise(test_image_tensor)
#Expand dims of the image to be able to feed it into the NN
test_image_tensor = tf.expand_dims(test_image_tensor, axis = 0)
#Get the model which is used to generate a denoised image
trained_model = my_autoencoder.get_model()
#Generated the denoised image
generated_image = trained_model(test_image_tensor)

#Plot that image
images_list = [test_image_tensor[0], generated_image[0]]
ih.display_images_together(images_list=images_list)





