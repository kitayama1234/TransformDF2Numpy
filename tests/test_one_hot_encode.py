import unittest
import numpy as np
import pandas as pd
import df2numpy
from df2numpy import TransformDF2Numpy, one_hot_encode, NAN_CATEGORY, DROPPED_CATEGORY
from df2numpy.errors import *


df = pd.DataFrame({
    "A": ["Aa", "Ab", "Ac", "Aa", "Ac", "Aa", "Aa", "Aa"],  # uniques: 3, to_be_thresholded: "Ab"
    "B": [1., -3., 0., 2, 3, 0, -1.3, 0.192],
    "C": ["Ca", np.nan, "Cc", "Ca", "Cc", "Ca", "Cc", "Cc"],  # uniques: 2, nan: 1
    "D": ["Da", "Db", "Dc", "Db", "Dc", "Da", np.nan, "Dc"],  # uniques: 3, nan: 1
    "E": [1., -3., np.nan, 2, np.nan, 0, -16.9, 20],
    "Drop": ["x", "x", "x", "x", "x", "x", "x", "x"],  # must be dropped
    "F": ["Fa", "Fb", "Fc", "Fd", "Fa", "Fb", "Fc", "Fd"],  # uniques: 4
})

test_df = pd.DataFrame({
    "A": ["Ac", "Aa"],
    "B": [1.4, 0.],
    "C": ["Cc", "Ca"],
    "D": ["Dc", "Db"],
    "E": [4.3, 2],
    "Drop": ["x", "x"],
    "F": ["Fd", "Fc"]
})

test_df_only1data = pd.DataFrame({
    "A": ["Ac"],
    "B": [1.4],
    "C": ["Cc"],
    "D": ["Dc"],
    "E": [4.3],
    "Drop": ["x"],
    "F": ["Fd"]
})

test_df_with_nan = pd.DataFrame({
    "A": ["Ac", np.nan],
    "B": [np.nan, 1.4],
    "C": [np.nan, "Cc"],
    "D": ["Dc", np.nan],
    "E": [4.3, np.nan],
    "Drop": ["x", np.nan],
    "F": [np.nan, "Fd"]
})

test_df_with_new_category = pd.DataFrame({
    "A": ["Ac", "Anew"],  # should be in DROPPED_CATEGORY
    "B": [1.4, 0.],
    "C": ["Cc", "Ca"],
    "D": ["Dnew", "Db"],  # should be in NAN_CATEGORY
    "E": [4.3, 2],
    "Drop": ["x", "x"],
    "F": ["Fd", "Fnew"]  # should be in the most frequent category 'Fd'
})

test_df_wrong_const1 = pd.DataFrame({
    "A": ["Ac", "Aa"],
    "B": [1.4, 0.],
    "Wrong": ["wtf", "???"],
    "D": ["Dc", "Db"],
    "E": [4.3, 2],
    "Drop": ["x", "x"],
    "F": ["Fd", "Fc"]
})

test_df_wrong_const2 = pd.DataFrame({
    "A": ["Ac", "Aa"],
    "C": ["Cc", "Ca"],
    "B": [1.4, 0.],
    "D": ["Dc", "Db"],
    "E": [4.3, 2],
    "Drop": ["x", "x"],
    "F": ["Fd", "Fc"]
})

test_df_wrong_const3 = pd.DataFrame({
    "A": ["Ac", "Aa"],
    "B": [1.4, 0.],
    "D": ["Dc", "Db"],
    "E": [4.3, 2],
    "Drop": ["x", "x"],
    "F": ["Fd", "Fc"]
})


class TestOneHotEncode(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_one_hot_encode_scaled(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True,
                              fillnan=True,
                              objective_col="B")

        x, y = t.fit_transform(df)

        x_one_hot, var_names = one_hot_encode(t, x)

        self.assertTrue(x_one_hot.shape == (8, 13))

        self.assertListEqual(var_names, ['A_Aa', 'A_TransformDF2Numpy_dropped_category', 'A_Ac', 'D_Da',
                                         'D_Db', 'D_Dc', 'D_TransformDF2Numpy_NaN_category', 'F_Fa', 'F_Fb',
                                         'F_Fc', 'F_Fd', 'C', 'E'])

        for i, name in enumerate(var_names):
            self.assertTrue(-0.00001 < x_one_hot[:, i].mean() < 0.00001)
            self.assertTrue(0.9999 < x_one_hot[:, i].std() < 1.00001)

    def test_one_hot_encode_fillnan_false(self):
        t = TransformDF2Numpy(min_category_count=2,
                              fillnan=False,
                              objective_col="B")

        x, y = t.fit_transform(df)

        x_one_hot, var_names = one_hot_encode(t, x)

        self.assertListEqual(var_names, ['A_Aa', 'A_TransformDF2Numpy_dropped_category', 'A_Ac', 'D_Da', 'D_Db',
                                         'D_Dc', 'F_Fa', 'F_Fb', 'F_Fc', 'F_Fd', 'C', 'E'])

        self.assertTrue(x_one_hot.shape == (8, 12))

        self.assertListEqual(list(x_one_hot[6, 3:6]), [0., 0., 0.])

    def test_one_hot_encode_eliminate_verbose_feature(self):
        t = TransformDF2Numpy(min_category_count=2,
                              fillnan=False,
                              objective_col="B")

        x, y = t.fit_transform(df)

        x_one_hot, var_names = one_hot_encode(t, x, elim_verbos=True)

        self.assertListEqual(var_names, ['A_Aa', 'A_TransformDF2Numpy_dropped_category', 'D_Da', 'D_Db',
                                         'F_Fa', 'F_Fb', 'F_Fc', 'C', 'E'])

        self.assertTrue(x_one_hot.shape == (8, 9))








