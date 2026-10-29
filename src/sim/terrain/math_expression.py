import re
import torch


def math_expression(world, exp):
    data_dict = {fk: world["data"][i] for i, fk in enumerate(world["feature_keys"])}

    res = eval(exp, {}, data_dict)
    return res
