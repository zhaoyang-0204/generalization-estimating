import numpy as np
import os


CHANCE_LEVEL = 0.01
UNIT_NUMBER = 512
CLASS_NUM = 100



def key_quantities(params,
                   use_nomalization_in_area = False,
                   layer_name = "block5_conv3_relu"):
    res_path = params["results_params"]["result_save_dir"]
    print("Calculate Key Quantities for %s...."%res_path)
    res = []
    for i in range(CLASS_NUM):
        f_path = os.path.join(res_path, "class_%d"%(i))
        curve_ablation = np.loadtxt(os.path.join(f_path, layer_name + "_0_ca_curves.txt"), delimiter=",")
        curve_insertion = np.loadtxt(os.path.join(f_path, layer_name + "_1_ca_curves.txt"), delimiter=",")

        n0 = find_turning_point(curve_ablation)
        n0_r = find_turning_point(curve_insertion, ablation=False)

        unit_sparse_rate = (n0 + n0_r) / (2 * UNIT_NUMBER)
        unit_sparse_area_ = np.average(curve_insertion[:] - curve_ablation[:], axis = 0)
        if use_nomalization_in_area:
            unit_sparse_area_ = unit_sparse_area_ / curve_insertion[0]

        res.append((unit_sparse_rate, unit_sparse_area_))

    unit_sparse_rate_average = np.average(res, axis =  0)
    return unit_sparse_rate_average


def find_turning_point(curve, ablation = True):
    
    target_curve = curve[:]
    if ablation:
        bool_position = target_curve <= CHANCE_LEVEL
        critical_point_list = np.where(bool_position == True)[0]
        if len(critical_point_list) != 0:
            critical_point = critical_point_list[0]
        else:
            critical_point = np.argmin(target_curve)

    else:
        invert_curve = target_curve[::-1][1:]
        critical_point = np.argmax(invert_curve) + 1
        
    return critical_point

def load_generalization_gap(params):
    model_path = params["model_params"]["save_dir"]
    log_path = os.path.join(model_path, "log.csv")
    gen_log = np.loadtxt(log_path, skiprows = 1, delimiter=",")
    gen_gap = gen_log[-1, 1] - gen_log[-1, -3]
    return gen_gap