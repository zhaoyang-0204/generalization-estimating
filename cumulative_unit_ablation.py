from model import VGG16Model, dataset_by_classes, load_model_params
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import numpy as np
import tensorflow as tf



def cumulative_unit_ablation(params):

    layer_name = "block5_conv3_relu"
    res_save_dir = params["results_params"]["result_save_dir"]

    vggModel = VGG16Model(**params["model_params"])
    vggModel_trained_model_path = os.path.join(params["model_params"]["save_dir"], "weights.hdf5")
    print("Loading trained model weights from %s"%vggModel_trained_model_path)
    vggModel.init_model(vggModel_trained_model_path, layer_name=layer_name)

    data_train_list, data_test_list = dataset_by_classes(os.path.join(params["data_params"]["save_random_data_dir"], "random_trained_label.pkl"))

    for class_i in range(100):

        if not os.path.exists(os.path.join(res_save_dir, "class_%d"%(class_i))):
            os.mkdir(os.path.join(res_save_dir, "class_%d"%(class_i)))
        
        x_train = data_train_list[class_i]
        np.random.shuffle(x_train)
        y_train = tf.keras.utils.to_categorical(np.repeat((class_i), x_train.shape[0]), 100)

        layer_outputs = vggModel.intermediate_layer_model.predict(x_train)
        average_arr, importance_ = average_mean_method(layer_outputs)
        importance_reverse = importance_.copy()
        importance_reverse.reverse()

        for type_i, importance_rank in enumerate([importance_, importance_reverse]):
            evaluations = vggModel.predict_model.evaluate(layer_outputs, y_train, batch_size = params["data_params"]["batch_size"], verbose = 1)
            unit_mask = np.ones((512,))
            evaluation_list = []
            evaluation_list.append(evaluations)

            for i, item in enumerate(importance_rank):
                unit_mask[importance_rank[i]] = 0
                feature_map = mask_layer_outputs(unit_mask, layer_outputs)
                evaluations = vggModel.predict_model.evaluate(feature_map, y_train, batch_size = params["data_params"]["batch_size"], verbose = 1)
                evaluation_list.append(evaluations)
            
            evaluation_list = np.array(evaluation_list)
            save_results_fname_final = os.path.join(params["data_params"]["save_random_data_dir"], "class_%d"%(class_i), layer_name + "_%d_ca_curves.txt"%(type_i))
            np.savetxt(save_results_fname_final, evaluation_list[:, 1], fmt="%.6f", delimiter="\t")



def average_mean_method(layer_output):
    """
        Important list is ranked by the averaged mean of the feature maps.
    """

    mean_arr = np.zeros(shape = (layer_output.shape[-1]))
    layer_output = layer_output.reshape((-1, layer_output.shape[-1]))
    for i in range(layer_output.shape[-1]):
        layer_output_slice = layer_output[:, i]
        mean_value = np.mean(layer_output_slice)
        mean_arr[i] = mean_value

    importantList = np.argsort(mean_arr)[::-1]
    importantList = importantList.tolist()

    return mean_arr, importantList


def mask_layer_outputs(unit_mask, layer_outputs):
    unit_mask_tensor = tf.constant(unit_mask, dtype = "float32")
    feature_map = layer_outputs * unit_mask_tensor
    return feature_map
