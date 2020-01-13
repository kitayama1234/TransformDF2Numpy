import numpy as np
import warnings
from .errors import *


def one_hot_encode(transformer, data):
    names = []
    for index, c_var in enumerate(transformer.categorical_variables()):
        temp = np.identity(transformer.nunique(index))[list(data[:, index].astype(np.int))]
        # nan handling
        if (data[:, index] == -1).sum():
            drop_target_indices = list(np.where(data[:, index] == -1)[0])
            temp[drop_target_indices, -1] = 0.

        # scaling
        if transformer.numerical_scaling:
            fac = transformer._get_transform(index)
            temp = (temp - fac.categories_one_hot_means) / fac.categories_one_hot_stds

        # create variable names like ["variable_category1", "variable_category2", ...]
        heads = np.array([transformer.index_to_name(index) + "_"] * len(transformer.categories(index)))
        names += list(np.core.defchararray.add(heads, list(transformer.categories(index))))

        # make one-hot array
        if index == 0:
            one_hot_array = temp
        else:
            one_hot_array = np.concatenate([temp, one_hot_array], axis=1)

    out = np.concatenate([one_hot_array, data[:, transformer.num_categoricals:]], axis=1)
    out_names = names + transformer.numerical_variables()

    return out, out_names

