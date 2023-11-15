from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf
import os

class VGG16Model():

    def __init__(self,
                 save_dir = None,
                 initial_epoch = 0,
                 dropout_list = [0.0, 0.0, 0.0],
                 conv_bn = False,
                 momentum = 0.9,
                 nesterov = True,
                 weight_decay = 5e-4,
                 epoch = 400,
                 batch_size = 128,
                 input_shape = (32, 32, 3)):


        self.batch_size = batch_size
        self.dropout_list = dropout_list
        self.conv_bn = conv_bn
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            try:
                os.mkdir(self.save_dir)
            except Exception as e:
                raise ("Fail to make model folder....")

        self.initial_epoch = initial_epoch
        self.epochs = epoch
        self.input_shape = input_shape

        self.model = self.build_model()


    def build_model(self):
        model = Sequential()
        #Block 1
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block1_conv1', input_shape = self.input_shape))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block1_conv1_relu'))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block1_conv2'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block1_conv2_relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        #Block 2
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block2_conv1'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block2_conv1_relu'))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block2_conv2'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block2_conv2_relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
        
        #Block 3
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block3_conv1'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block3_conv1_relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block3_conv2'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block3_conv2_relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block3_conv3'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block3_conv3_relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        #Block 4
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block4_conv1'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block4_conv1_relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block4_conv2'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block4_conv2_relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block4_conv3'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block4_conv3_relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        
        #Block 5
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block5_conv1'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block5_conv1_relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block5_conv2'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block5_conv2_relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='block5_conv3'))
        if self.conv_bn: model.add(BatchNormalization())
        model.add(Activation("relu", name='block5_conv3_relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        #Fully connected
        model.add(Flatten(name="flatten"))
        model.add(Dropout(self.dropout_list[0]))
        model.add(Dense(4096, kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='fc1'))
        model.add(Activation("relu", name='fc1_relu'))
        model.add(Dropout(self.dropout_list[1]))
        model.add(Dense(4096, kernel_regularizer=regularizers.l2(self.weight_decay), bias_regularizer=regularizers.l2(self.weight_decay), name='fc2'))
        model.add(Activation("relu", name='fc2_relu'))
        model.add(Dropout(self.dropout_list[2]))

        model.add(Dense(100, activation="softmax", name='predictions'))
        return model

    def train_model(self, train_set, validation_set):

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        
        self.model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.SGD(learning_rate=1e-2, momentum=self.momentum, nesterov=self.nesterov), metrics=["acc", tf.keras.metrics.top_k_categorical_accuracy])
           
        training_optimizer = tf.keras.callbacks.ReduceLROnPlateau("loss", patience=5, min_delta=1e-4)
        stop_callback = haltCallback()
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.save_dir, "log.csv"),  append=True)

        self.model.fit(train_set, epochs = self.epochs, initial_epoch=self.initial_epoch, validation_data = validation_set, 
                    validation_freq= 1, batch_size=self.batch_size, callbacks=[stop_callback, training_optimizer, csv_logger])
 
        self.model.save_weights(os.path.join(self.save_dir, "weights.hdf5"))


    def build_intermediate_model(self, layer_name = "block5_conv3_relu"):
        self.intermediate_layer_model = tf.keras.models.Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)

    def build_predict_model(self, layer_name = "block5_conv3_relu"):
        target_layer_index = -1
        for i, l in enumerate(self.model.layers):
            if l.name == layer_name:
                target_layer_index = i
                input_shape = l.output_shape[1:]
        if target_layer_index == -1:
            raise Exception("Layer name not found!")

        inputs = tf.keras.layers.Input(input_shape)
        x = self.model.layers[target_layer_index + 1](inputs)
        for l in self.model.layers[target_layer_index + 2::]:
            x = l(x)
    
        self.predict_model = tf.keras.models.Model(inputs, x, name="predict_model")


    def init_model(self, fname, layer_name = "block5_conv3"):
        self.model.load_weights(fname)
        self.model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc", tf.keras.metrics.top_k_categorical_accuracy])
        self.build_intermediate_model(layer_name=layer_name)
        self.build_predict_model(layer_name = layer_name)
        self.predict_model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc", tf.keras.metrics.top_k_categorical_accuracy])


class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(self.model.optimizer.lr <= 1.1e-5):
            print("\n\nLearning rate reaches below threshold, training terminates!\n\n")
            self.model.stop_training = True

        print(logs.get('acc'))
        if(logs.get('acc') >= 0.97):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True


if __name__ == "__main__":
    VGG16Model().build_model()  