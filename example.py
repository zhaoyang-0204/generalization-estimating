import model
from cumulative_unit_ablation import cumulative_unit_ablation
from calculate_key_quantities import key_quantities, load_generalization_gap
from generalization_estimation import generalization_gap_estimation
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """
        An example of the whole implementation, including training, cumulative
        unit ablation and estimating. 

        All the calculation depends on the parameters configured in the json
        file in the model directory. 

        If you do not want to train the model, you could assign the training
        flag to False. But note that the same configure in the json should still
        be filled for further estimating.

    """


    Training_Flag = False
    Ablation = False

    # Load the parameters in the json file
    params = model.load_model_params()

    # Train all the models listed in the json file, and you should configure the
    # model in the json file before training. If you have trained networks, you
    # should omitted this part by assigning the flag to False.
    if Training_Flag:
        for keys, values in params.items():
            print(values["decription"])
            model.model_training(values)


    # This calculates the cumulative unit ablation curves for all the classes.
    # This may cost hours for calculating on the whole dataset. Certainly, you
    # could change to only calculate part of the classes, where classes are
    # recommended to be selected randomly.
    if Ablation: 
        for keys, values in params.items():
            print(values["decription"])
            cumulative_unit_ablation(values)


    # This calculates the key quantities. This should be done after performing
    # cumulative unit ablation.
    quantity_collection = []
    for keys, values in params.items():
        quantity1, quantity2 = key_quantities(values)
        # This load the generalization gap. Make sure the log.csv file is in your model directory.
        generalization_gap = load_generalization_gap(values)
        quantity_collection.append((quantity1, quantity2, generalization_gap))
    
    print("Key quantities and generalization gap are:")
    print(np.array(quantity_collection))
    generalization_gap_estimation(np.array(quantity_collection))


if __name__ == "__main__":
    main()