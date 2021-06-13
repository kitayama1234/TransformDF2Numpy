import unittest
import numpy as np
import pandas as pd
import df2numpy
from df2numpy import TransformDF2Numpy, NAN_CATEGORY, DROPPED_CATEGORY
from df2numpy.errors import *


df2numpy.modules.logging = False


# method test
class TestTransformDF2Numpy(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "A": ["Aa", "Ab", "Ac", "Aa", "Ac", "Aa", "Aa", "Aa"],  # uniques: 3, to_be_thresholded: "Ab"
            "B": [1., -3., 0., 2, 3, 0, -1.3, 0.192],
            "C": ["Ca", np.nan, "Cc", "Ca", "Cc", "Ca", "Cc", "Cc"],  # uniques: 2, nan: 1
            "D": ["Da", "Db", "Dc", "Db", "Dc", "Da", np.nan, "Dc"],  # uniques: 3, nan: 1
            "E": [1., -3., np.nan, 2, np.nan, 0, -16.9, 20],
            "Drop": ["x", "x", "x", "x", "x", "x", "x", "x"],  # must be dropped
            "F": ["Fa", "Fb", "Fc", "Fd", "Fa", "Fb", "Fc", "Fd"],  # uniques: 4
        })

        self.test_df = pd.DataFrame({
            "A": ["Ac", "Aa"],
            "B": [1.4, 0.],
            "C": ["Cc", "Ca"],
            "D": ["Dc", "Db"],
            "E": [4.3, 2],
            "Drop": ["x", "x"],
            "F": ["Fd", "Fc"]
        })

        self.test_df_only1data = pd.DataFrame({
            "A": ["Ac"],
            "B": [1.4],
            "C": ["Cc"],
            "D": ["Dc"],
            "E": [4.3],
            "Drop": ["x"],
            "F": ["Fd"]
        })

        self.test_df_with_nan = pd.DataFrame({
            "A": ["Ac", np.nan],
            "B": [np.nan, 1.4],
            "C": [np.nan, "Cc"],
            "D": ["Dc", np.nan],
            "E": [4.3, np.nan],
            "Drop": ["x", np.nan],
            "F": [np.nan, "Fd"]
        })

        self.test_df_with_new_category = pd.DataFrame({
            "A": ["Ac", "Anew"],  # should be in DROPPED_CATEGORY
            "B": [1.4, 0.],
            "C": ["Cc", "Ca"],
            "D": ["Dnew", "Db"],  # should be in NAN_CATEGORY
            "E": [4.3, 2],
            "Drop": ["x", "x"],
            "F": ["Fd", "Fnew"]  # should be in the most frequent category 'Fd'
        })

        self.test_df_wrong_const1 = pd.DataFrame({
            "A": ["Ac", "Aa"],
            "B": [1.4, 0.],
            "Wrong": ["wtf", "???"],
            "D": ["Dc", "Db"],
            "E": [4.3, 2],
            "Drop": ["x", "x"],
            "F": ["Fd", "Fc"]
        })

        self.test_df_wrong_const2 = pd.DataFrame({
            "A": ["Ac", "Aa"],
            "C": ["Cc", "Ca"],
            "B": [1.4, 0.],
            "D": ["Dc", "Db"],
            "E": [4.3, 2],
            "Drop": ["x", "x"],
            "F": ["Fd", "Fc"]
        })

        self.test_df_wrong_const3 = pd.DataFrame({
            "A": ["Ac", "Aa"],
            "B": [1.4, 0.],
            "D": ["Dc", "Db"],
            "E": [4.3, 2],
            "Drop": ["x", "x"],
            "F": ["Fd", "Fc"]
        })

    def test_instance_creation(self):
        # invalid input for specifying objective col
        with self.assertRaises(InvalidInputForSpecifyingObjectiveColumnError):
            TransformDF2Numpy(objective_col=4)

    def test_fit_transform_basic(self):
        t = TransformDF2Numpy()

        x = t.fit_transform(self.df)

        # size
        self.assertTrue(x.shape == (8, 6))

        # output check
        self.assertListEqual(list(x[:, 0]), [0., 1., 2., 0., 2., 0., 0., 0.])
        self.assertListEqual(list(x[:, 1]), [0., 1., 2., 1., 2., 0., 3., 2.])
        self.assertListEqual(list(x[:, 2]), [0., 1., 2., 3., 0., 1., 2., 3.])
        self.assertListEqual(list(np.round(x[:, 3], decimals=3)), [1., -3., 0., 2., 3., 0., -1.3, 0.192])
        self.assertListEqual(list(np.round(x[:, 4], decimals=8)), [0., 0.57142857, 1., 0., 1., 0., 1., 1.])
        self.assertListEqual(list(np.round(x[:, 5], decimals=5)), [1., -3., 0.51667, 2., 0.51667, 0., -16.9, 20.])

    def test_fit_transform_category_threshold(self):
        t = TransformDF2Numpy(min_category_count=2)

        x = t.fit_transform(self.df)

        # size
        self.assertTrue(x.shape == (8, 6))

        # output check
        self.assertListEqual(list(x[:, 0]), [0., 1., 2., 0., 2., 0., 0., 0.])
        self.assertListEqual(list(x[:, 1]), [0., 1., 2., 1., 2., 0., 3., 2.])
        self.assertListEqual(list(x[:, 2]), [0., 1., 2., 3., 0., 1., 2., 3.])
        self.assertListEqual(list(np.round(x[:, 3], decimals=3)), [1., -3., 0., 2., 3., 0., -1.3, 0.192])
        self.assertListEqual(list(np.round(x[:, 4], decimals=8)), [0., 0.57142857, 1., 0., 1., 0., 1., 1.])
        self.assertListEqual(list(np.round(x[:, 5], decimals=5)), [1., -3., 0.51667, 2., 0.51667, 0., -16.9, 20.])

    def test_fit_transform_fillnan_false(self):
        t = TransformDF2Numpy(min_category_count=2,
                              fillnan=False)

        x = t.fit_transform(self.df)

        # output check for nans
        self.assertTrue(x[6, 1] == -1.)     # category: -1
        self.assertTrue(np.isnan(x[1, 4]))  # numerical: nan
        self.assertTrue(np.isnan(x[2, 5]))
        self.assertTrue(np.isnan(x[4, 5]))

    def test_fit_transform_numerical_scaling(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)

        x = t.fit_transform(self.df)

        # numerical variable should be scaled
        self.assertTrue(-0.00001 < x[:, t.name_to_index("C")].mean() < 0.00001)
        self.assertTrue(0.9999 < x[:, t.name_to_index("C")].std() < 1.00001)
        self.assertTrue(-0.00001 < x[:, t.name_to_index("B")].mean() < 0.00001)
        self.assertTrue(0.9999 < x[:, t.name_to_index("B")].std() < 1.00001)
        self.assertTrue(-0.00001 < x[:, t.name_to_index("E")].mean() < 0.00001)
        self.assertTrue(0.9999 < x[:, t.name_to_index("E")].std() < 1.00001)

    def test_fit_transform_numerical_objective_col(self):
        t = TransformDF2Numpy(min_category_count=2,
                              objective_col="B")

        x, y = t.fit_transform(self.df)

        # x size
        self.assertTrue(x.shape == (8, 5))

        # check y output
        self.assertListEqual(list(np.round(y, decimals=3)), [1., -3., 0., 2., 3., 0., -1.3, 0.192])

    def test_fit_transform_numerical_objective_col_with_scaling(self):
        t = TransformDF2Numpy(min_category_count=2,
                              objective_col="B",
                              objective_scaling=True)

        x, y = t.fit_transform(self.df)

        # check y output (below has mean 0. and std 1.)
        self.assertListEqual(list(np.round(y, decimals=8)), [0.43826295, -1.85781013, -0.13575532, 1.01228122,
                                                             1.58629949, -0.13575532, -0.88197907, -0.02554381])

    def test_fit_transform_categorical_objective_col(self):
        t = TransformDF2Numpy(min_category_count=2,
                              objective_col="A")

        x, y = t.fit_transform(self.df)

        # x size
        self.assertTrue(x.shape == (8, 5))

        self.assertListEqual(list(y), ['Aa', 'Ab', 'Ac', 'Aa', 'Ac', 'Aa', 'Aa', 'Aa'])

    def test_fit_transform_categorical_objective_col_with_scaling(self):
        t = TransformDF2Numpy(min_category_count=2,
                              objective_col="A",
                              objective_scaling=True)

        x, y = t.fit_transform(self.df)

        # scaling flag is ignored
        self.assertListEqual(list(y), ['Aa', 'Ab', 'Ac', 'Aa', 'Ac', 'Aa', 'Aa', 'Aa'])

    def test_transform(self):
        t = TransformDF2Numpy(min_category_count=2,
                              objective_col="B",
                              objective_scaling=True)

        x, y = t.fit_transform(self.df)

        x_test, y_test = t.transform(self.test_df)
        self.assertTrue((x_test == np.array([[2., 2., 3., 1., 4.3], [0., 1., 2., 0., 2.]])).all())
        self.assertTrue((np.round(y_test, decimals=8) == np.array([0.66787026, -0.13575532])).all())

        x_test_only1, y_test_only1 = t.transform(self.test_df_only1data)
        self.assertTrue((x_test_only1 == np.array([[2., 2., 3., 1., 4.3]])).all())
        self.assertTrue((np.round(y_test_only1, decimals=8) == np.array([0.66787026])).all())

        x_test_with_nan, y_test_with_nan = t.transform(self.test_df_with_nan)
        self.assertTrue((np.round(x_test_with_nan, decimals=8) == np.array([[2., 2., 0., 0.57142857, 4.3],
                                                                            [0., 3., 3., 1., 0.51666667]])).all())
        self.assertTrue(np.isnan(y_test_with_nan[0]))

        x_test_with_new_category, y_test_with_new_category = t.transform(self.test_df_with_new_category)
        self.assertTrue((x_test_with_new_category == np.array([[2., -1., 3., 1., 4.3], [1., 1., -1., 0., 2.]])).all())
        self.assertTrue((np.round(y_test_with_new_category, decimals=8) == np.array([0.66787026, -0.13575532])).all())

        # wrong inputs
        with self.assertRaises(WrongDataFrameConstructionError):
            t.transform(self.test_df_wrong_const1)

        with self.assertRaises(WrongDataFrameConstructionError):
            t.transform(self.test_df_wrong_const2)

        with self.assertRaises(WrongDataFrameConstructionError):
            t.transform(self.test_df_wrong_const3)

    # TODO: variables(self)
    def test_variables(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        self.assertListEqual(t.variables(), ['A', 'D', 'F', 'B', 'C_Cc', 'E'])

        t2 = TransformDF2Numpy(min_category_count=2,
                               objective_col="B")
        x2, y2 = t2.fit_transform(self.df)
        self.assertListEqual(t2.variables(), ['A', 'D', 'F', 'C_Cc', 'E'])

    # TODO: categoricals(self)
    def test_categoricals(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        self.assertListEqual(t.categoricals(), ['A', 'D', 'F'])

        t2 = TransformDF2Numpy(min_category_count=2,
                               objective_col="B")
        x2, y2 = t2.fit_transform(self.df)
        self.assertListEqual(t2.categoricals(), ['A', 'D', 'F'])

    # TODO: numericals(self)
    def test_numericals(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        self.assertListEqual(t.numericals(), ['B', 'C_Cc', 'E'])

        t2 = TransformDF2Numpy(min_category_count=2,
                               objective_col="B")
        x2, y2 = t2.fit_transform(self.df)
        self.assertListEqual(t2.numericals(), ['C_Cc', 'E'])

    # TODO: name_to_index(self, colname)
    # VariableNotExistError
    def test_name_to_index(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        for index, name in enumerate(['A', 'D', 'F', 'B', 'C', 'E']):
            self.assertEqual(index, t.name_to_index(name))
        with self.assertRaises(VariableNotExistError):
            t.name_to_index("Drop")
        with self.assertRaises(VariableNotExistError):
            t.name_to_index(3)

        t2 = TransformDF2Numpy(min_category_count=2,
                               objective_col="B")
        x2, y2 = t2.fit_transform(self.df)
        for index, name in enumerate(['A', 'D', 'F', 'C', 'E']):
            self.assertEqual(index, t2.name_to_index(name))
        with self.assertRaises(VariableNotExistError):
            t2.name_to_index("Drop")
        with self.assertRaises(VariableNotExistError):
            t2.name_to_index(3)

    # TODO: index_to_name(self, index)
    def test_index_to_name(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        for index, name in enumerate(['A', 'D', 'F', 'B', 'C', 'E']):
            self.assertEqual(t.index_to_name(index), name)

        t2 = TransformDF2Numpy(min_category_count=2,
                               objective_col="B")
        x2, y2 = t2.fit_transform(self.df)
        for index, name in enumerate(['A', 'D', 'F', 'C', 'E']):
            self.assertEqual(t2.index_to_name(index), name)

    # TODO: is_numerical(self, index_or_colname)
    # VariableNotExistError
    # InvalidInputForSpecifyingVariableError
    def test_is_numerical(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        for index, name in enumerate(['A', 'D', 'F']):
            self.assertFalse(t.is_numerical(index))
            self.assertFalse(t.is_numerical(name))
        for index in [3, 4, 5]:
            self.assertTrue(t.is_numerical(index))
        for name in ['B', 'C', 'E']:
            self.assertTrue(t.is_numerical(name))
        with self.assertRaises(VariableNotExistError):
            t.is_numerical("Drop")
        with self.assertRaises(InvalidInputForSpecifyingVariableError):
            t.is_numerical(["B"])

        t2 = TransformDF2Numpy(min_category_count=2,
                               objective_col="B")
        x2, y2 = t2.fit_transform(self.df)
        for index, name in enumerate(['A', 'D', 'F']):
            self.assertFalse(t2.is_numerical(index))
            self.assertFalse(t.is_numerical(name))
        for index in [3, 4]:
            self.assertTrue(t2.is_numerical(index))
        for name in ['C', 'E']:
            self.assertTrue(t2.is_numerical(name))
        with self.assertRaises(VariableNotExistError):
            t2.is_numerical("45")
        with self.assertRaises(InvalidInputForSpecifyingVariableError):
            t.is_numerical(1.)

    # TODO: categories
    def test_categories(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)

        # invalid input type 1: float
        with self.assertRaises(InvalidInputForSpecifyingVariableError):
            t.categories(0.)

        # invalid input type 2: list
        with self.assertRaises(InvalidInputForSpecifyingVariableError):
            t.categories([0])

        # the variable has no categories
        with self.assertRaises(HasNoDictionaryError):
            t.categories(5)
        with self.assertRaises(HasNoDictionaryError):
            t.categories("B")

        # output check
        self.assertListEqual(list(t.categories(0)), ['Aa', DROPPED_CATEGORY, 'Ac'])
        self.assertListEqual(list(t.categories("D")), ['Da', 'Db', 'Dc', NAN_CATEGORY])
        self.assertListEqual(list(t.categories("C")), ['Ca', 'Cc'])
        self.assertListEqual(list(t.categories("F")), ['Fa', 'Fb', 'Fc', 'Fd'])

    # TODO: category_to_factorized(self, index_or_colname, category_name):
    # CategoryNotExistError
    # VariableNotExistError
    def test_category_to_factorized(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        for factorized, category in enumerate(['Aa', DROPPED_CATEGORY, 'Ac']):
            self.assertTrue(factorized == t.category_to_factorized(0, category))
        for factorized, category in enumerate(['Da', 'Db', 'Dc', NAN_CATEGORY]):
            self.assertTrue(factorized == t.category_to_factorized("D", category))
        for factorized, category in enumerate(['Fa', 'Fb', 'Fc', 'Fd']):
            self.assertTrue(factorized == t.category_to_factorized("F", category))
        self.assertTrue(0.925818099776872 == t.category_to_factorized("C", "Cc"))
        self.assertTrue(-1.2344241330358292 == t.category_to_factorized("C", "Ca"))
        with self.assertRaises(CategoryNotExistError):
            t.category_to_factorized("C", "Cb")
        with self.assertRaises(VariableNotExistError):
            t.category_to_factorized("Drop", "x")

        t2 = TransformDF2Numpy()
        x2 = t2.fit_transform(self.df)
        for factorized, category in enumerate(['Aa', 'Ab', 'Ac']):
            self.assertTrue(factorized == t2.category_to_factorized(0, category))

    # TODO: factorized_to_category(self, index_or_colname, factorized_value):
    # FactorizedNotExistError
    def test_factorized_to_category(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        for factorized, category in enumerate(['Aa', DROPPED_CATEGORY, 'Ac']):
            self.assertTrue(t.factorized_to_category(0, factorized) == category)
        for factorized, category in enumerate(['Da', 'Db', 'Dc', NAN_CATEGORY]):
            self.assertTrue(t.factorized_to_category("D", factorized) == category)
        for factorized, category in enumerate(['Fa', 'Fb', 'Fc', 'Fd']):
            self.assertTrue(t.factorized_to_category("F", factorized) == category)
        self.assertTrue(t.factorized_to_category("C", 0.925818099776872) == "Cc")
        self.assertTrue(t.factorized_to_category("C", -1.2344241330358292) == "Ca")
        with self.assertRaises(FactorizedNotExistError):
            t.factorized_to_category("A", 3)

        t2 = TransformDF2Numpy()
        x2 = t2.fit_transform(self.df)
        for factorized, category in enumerate(['Aa', 'Ab', 'Ac']):
            self.assertTrue(t2.factorized_to_category(0, factorized) == category)

    # TODO: nuniques(self)
    def test_nuniques(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        self.assertListEqual(t.nuniques(), [3, 4, 4])

    # TODO: nunique(self, index_or_colname)
    # WronglySpecifiedNumericalVariableError
    def test_nunique(self):
        t = TransformDF2Numpy(min_category_count=2,
                              numerical_scaling=True)
        x = t.fit_transform(self.df)
        self.assertListEqual(t.nuniques(), [3, 4, 4])

        correct_uniques = [3, 4, 4]
        for index, name in enumerate(['A', 'D', 'F']):
            self.assertTrue(t.nunique(index) == correct_uniques[index])
            self.assertTrue(t.nunique(name) == correct_uniques[index])

        with self.assertRaises(WronglySpecifiedNumericalVariableError):
            t.nunique("B")


if __name__ == '__main__':
    unittest.main()

