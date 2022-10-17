import tensorflow as tf
import matplotlib.pyplot as plt

def load_image_label(image_path, target_size):
    image_tensor = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_jpeg(image_tensor)
    image_tensor = tf.cast(image_tensor, dtype = tf.float32)

    #Scale the image
    image_tensor = image_tensor/255

    #Crop the image
    image_shape = image_tensor.shape
    dim1 = image_shape[0]
    dim2 = image_shape[1]
    min_dim = min(dim1, dim2)
    offset = int(min_dim/2)
    center1 = int(dim1/2)
    center2 = int(dim2/2)

    dim1_0=center1-offset
    dim1_1=center1+offset
    dim2_0=center2-offset
    dim2_1=center2+offset
    image_tensor = image_tensor[dim1_0:dim1_1, dim2_0:dim2_1, :]

    #Resize to a target size
    image_tensor = tf.image.resize(image_tensor, target_size)

    return image_tensor


def add_synthetic_noise(image_tensor):
    noise_factor = 0.02
    factor = noise_factor * tf.random.normal(shape=image_tensor.shape)
    image_noisy = image_tensor + factor
    image_noisy = tf.clip_by_value(image_noisy, 0.0, 1.0)
    return image_noisy

def display_image(image_tensor):
    plt.imshow(image_tensor)
    plt.show()
    pass


def display_images_together(images_list):
    fig = plt.figure(figsize=(15, 15))
    rows = 1
    columns = len(images_list)

    fig.add_subplot(rows, columns, 1)
    plt.imshow(images_list[0])
    plt.title("Noisy Image")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(images_list[1])
    plt.title("De-noised Image")
    plt.show()
    pass