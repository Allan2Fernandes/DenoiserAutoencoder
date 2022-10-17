import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot
from keras.layers import LeakyReLU, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras.losses import MeanSquaredError


class Autoencoder_model:
    def __init__(self, input_shape, train_data_generator, validation_data_generator):
        self.input_shape = input_shape
        self.train_data_generator = train_data_generator
        self.validation_data_generator = validation_data_generator
        self.filters = 64
        pass

    def get_encoder(self, input_layer):
        x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(5,5), padding='same')(input_layer)
        x = LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)
        #x = BatchNormalization()(x)
        skip_connection_1 = x
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x) #Size is down to 1/2

        x = tf.keras.layers.Conv2D(filters=self.filters*2, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=self.filters*2, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)
        #x = BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # Size is down to 1/4

        # x = tf.keras.layers.Conv2D(filters=self.filters*4, kernel_size=(3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.Conv2D(filters=self.filters*4, kernel_size=(3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # Size is down to 1/8
        #
        # x = tf.keras.layers.Conv2D(filters= 256, kernel_size=(3, 3), padding='same')(x)
        # x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # Size is down to 1/16
        #
        # x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
        # x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # Size is down to 1/32
        return x, skip_connection_1

    def bottleneck_layer(self, input_layer):
        bottle_neck = tf.keras.layers.Conv2D(filters=self.filters*4, kernel_size=(3, 3),  padding='same')(input_layer)
        bottle_neck = LeakyReLU()(bottle_neck)
        return bottle_neck

    def get_conv2d_transpose_decoder(self, input_layer, skip_connection_1):
        x = Conv2DTranspose(filters = self.filters*2, strides = (2,2), kernel_size=(3,3), padding = 'same')(input_layer)
        x = LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=self.filters*2, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        #x = BatchNormalization()(x)
        x = Conv2DTranspose(filters = self.filters, strides = (2,2), padding = 'same', kernel_size=(3,3))(x)
        x = LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        #x = BatchNormalization()(x)
        # x = Conv2DTranspose(filters=self.filters, strides=(2, 2), padding='same', kernel_size=(3, 3))(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.add([x, skip_connection_1])    #TEST WITHOUT THIS ONE FIRST
        x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
        return x


    def get_upsampling_decoder(self, input_layer, skip_connection_1):
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(input_layer)# Size is 2x
        x = tf.keras.layers.Conv2D(filters=self.filters*2, kernel_size=(5,5), padding='same')(x)
        x = LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=self.filters*2, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)

        #x = BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # Size is 4x
        x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)

        #x = BatchNormalization()(x)
        # x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # Size is 8x
        #
        # x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        # x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # Size is 16x
        #
        # x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        # x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        # x = LeakyReLU()(x)
        # x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # Size is 32x
        x = tf.keras.layers.add([x, skip_connection_1])
        #Final classifcation layer
        x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (5,5), activation = 'sigmoid', padding = 'same')(x)
        return x

    def build_whole_model(self):
        input_layer = tf.keras.layers.Input(shape = self.input_shape)
        encoder, skip_connection = self.get_encoder(input_layer)
        bottleneck = self.bottleneck_layer(encoder)
        decoder = self.get_upsampling_decoder(bottleneck, skip_connection)

        self.model = tf.keras.Model(inputs = input_layer, outputs = decoder)
        pass

    def show_model_summary(self):
        self.model.summary()
        pass


    def compile_model(self):
        loss = MeanSquaredError()
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss, metrics = ['accuracy'])
        pass

    def get_model(self):
        return self.model

    def train_model(self, epochs):
        history = self.model.fit(self.train_data_generator, epochs = epochs, validation_data = self.validation_data_generator)
        pass




