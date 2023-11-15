import os
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random


def load_data(fileURL):
    with open(fileURL, "rb") as handle:
        return pickle.load(handle)

def save_data(fileURL, data):
    with open(fileURL, "wb") as handle:
        pickle.dump(data, handle)    

def corrupt_labels(x, y, class_num = 100, random_rate = 0.0):
    print("Changing %.2f percent of labels"%(random_rate))
    sample_num = y.shape[0]
    random_num = int(sample_num * random_rate)
    choose_to_randomize = random.sample([i for i in range(sample_num)], random_num)
    for item in choose_to_randomize:
        y[item] = random.randint(0, class_num - 1)
    
    return (x, y)


def load_dataset(
                random_rate = 0.0,
                save_random_data_dir = None,
                load_random_data_dir = None,
                ):
    
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Randomize the traning label
    if load_random_data_dir:
        y_train = load_data(os.path.join(load_random_data_dir, "random_trained_label.pkl"))            
    else:
        x_train, y_train = corrupt_labels(x_train, y_train, class_num=100, random_rate = random_rate)
        if save_random_data_dir:
            save_data(os.path.join(save_random_data_dir, "random_trained_label.pkl"), y_train)

    return (x_train, y_train), (x_test, y_test)


def dataset_generator(
                        batch_size = 128,
                        horizontal_flip = True,
                        rotation_range = 15,
                        height_shift_range = 0.1,
                        width_shift_range = 0.1,
                        random_rate = 0.0,
                        save_random_data_dir = None,
                        load_random_data_dir = None,):

    (x_train, y_train), (x_test, y_test) = load_dataset(random_rate, save_random_data_dir, load_random_data_dir)
    # Generate dataset 
    ds_gen_train = ImageDataGenerator(preprocessing_function= preprocess_input, horizontal_flip=horizontal_flip, 
            rotation_range=rotation_range, height_shift_range=height_shift_range, width_shift_range=width_shift_range)
    
    ds_gen_train.fit(x_train)
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    train_set = ds_gen_train.flow(x_train, y_train, batch_size=batch_size)
    
    x_test = preprocess_input(x_test)
    y_test = tf.keras.utils.to_categorical(y_test, 100)
    validation_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    validation_set = validation_set.shuffle(x_test.shape[0]).batch(batch_size)

    return train_set, validation_set

def dataset_by_classes(random_label_fname = None):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    if random_label_fname:
        y_train = load_data(random_label_fname)

    data_train_list = []
    for i in range(100):
        data_train_list.append(x_train[y_train[:, 0] == i])

    data_test_list = []
    for i in range(100):
        data_test_list.append(x_test[y_test[:, 0] == i])

    return data_train_list, data_test_list


if __name__ == "__main__":
    load_dataset()