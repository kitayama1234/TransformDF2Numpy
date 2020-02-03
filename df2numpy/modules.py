"""
TransformDF2Numpy is a simple tool for quick transformation from pandas.DataFrame to numpy.array dataset,
containing some utilities such as re-transformation of new data,
minimal pre-processing, and access to variable information.


##################
###  Overview  ###
##################

    + Transform a training set of the pandas.DataFrame to a numpy.array dataset, and fit a transformer instance.
      The numpy.array containing the factorized categorical variables (first half)
      and the numerical variables (second half).
    
    + Utilities of a fitted transformer instance.
        + Transforming New DataFrame samely as DataFrame used for fitting.
        + Access to variable information.
            + linking variable index and name
            + variable names (all, categorical, numerical)
            + linking factorized value and category name
            + unique categories of categorical variables
    
    + Minimal pre-processing (optional).
        + Scaling numerical variables.
            + robustness control by a parameter
        + Thresholding categorical variables by minimum count of each variable.
        + Filling missing values.
            + new category (or the most frequent category) for categorical variables.
            + mean value for numerical variables
            + robustness control by a parameter

    (Note: A categorical variable which has only two unique categories is treated as a numerical variable)



(*) Factorization: The process of converting each element of a categorical variable into a corresponding positive index.


####################
###  Parameters  ###
####################

              objective_col   : str (optional, default None)
                                The column name of objective variable.
                                If you specify this, the instance automatically find the column
                                and the output numpy array will be splitted into
                                x (explanatory variables) and y (objective variables).

          objective_scaling   : bool (optional, default False)
                                The flag for scaling objective variable.

          numerical_scaling   : bool (optional, default False)
                                The flag for scaling numerical variables.

  scaling_robustness_factor   : float in range of [0. 1.] (optional, default 0.)
                                The parameter to control robustness of scaling operation.
                                Specifying a larger value will make it more robust against outliers.

                    fillnan   : bool (optional, default True)
                                The flag to fill missing values (nan, NaN).
                                If True, the numerical nan will be filled with the mean,
                                and the categorical nan will be filled as new category (or most frequent category).
                                If False, the numerical nan will not be filled,
                                and the categorical nan will be filled with -1.

  fillnan_robustness_factor   : float in range of [0. 1.] (optional, default 0.)
                                The parameter to control robustness of calculating the filling value to nan.
                                Specifying a larger value will make it more robust against outliers.

         min_category_count   : integer (optional, default 0)
                                The minimum number of appearance of each category, in each categorical variable.
                                The categories with a number of appearance below this parameter will be thresholded,
                                and treated as a new single category.

                       copy   : bool (optional, default True)
                                Set to False to perform inplace the input DataFrame and avoid a copy.


#################
###  Methods  ###
#################

    fit_transform(df)
             Inputs:   training set of DataFrame
            Returns:   x, (y)
                       x : The numpy.array containing factorized categorical variables (first half)
                           and numerical variables (second half).
                           The variables which have only two unique categories are treated as numerical variables.
                       y : numpy array of objective variable (returned only when objective column exists)

    transform(df)
             Inputs:   testing set of DataFrame
            Returns:   x, (y)
                       x : numpy array of explanatory variables same as fit_transform()
                       y : numpy array of objective variable (only when objective column exists)

    variables()
            Returns:  the list of the name of all variables in order of the output numpy array
    
    categoricals()
            Returns:  the list of the name of categorical variables in order of the output numpy array

    numericals()
            Returns:  the list of the name of numerical variables in order of the output numpy array

    name_to_index(colname)
             Inputs:   column name of DataFrame
            Returns:   the corresponding column index of numpy array

    index_to_name(index)
             Inputs:   column index of numpy array
            Returns:   the corresponding column name of DataFrame

    is_numerical(index_or_colname)
             Inputs:   column index of numpy array
            Returns:   the bool indicating whether the variable is treated as a numerical variable or not

    categories(index_or_colname)
             Inputs:   column name of DataFrame, or column index of numpy array
             Return:   the list of unique categories in the variable which index correspond to the factorized values

    category_to_factorized(index_or_colname, category_name):
             Inputs:     index_or_colname : column name of DataFrame, or column index of numpy array
                            category_name : name of the single category
            Returns:   the factorized value

    factorized_to_category(index_or_colname, factorized_value):
             Inputs:     index_or_colname : column name of DataFrame, or column index of numpy array
                         factorized_value : factorized value of the single category
            Returns:   the name of the single category

    nuniques()
            Returns:   the list of the number of unique categories of the categorical variables

    nunique(index_or_colname)
             Inputs:   column name of DataFrame, or column index of numpy array
            Returns:   the number of unique categories of the categorical variable


####################
###  Attributes  ###
####################

              self.y_mean   :  the mean of the objective variable before scaling
    
               self.y_std   :  the standard deviation of the objective variable before scaling
    
    self.num_categoricals   :  the number of the categorical variables
    
      self.num_numericals   :  the number of the numerical variables

"""

import pandas as pd
import numpy as np
import warnings
from .errors import *


# global parameters
logging = True

# global constants
DROPPED_CATEGORY = "TransformDF2Numpy_dropped_category"
NAN_CATEGORY = "TransformDF2Numpy_NaN_category"


class TransformDF2Numpy:
    def __init__(self,
                 objective_col=None,
                 objective_scaling=False,
                 numerical_scaling=False,
                 scaling_robustness_factor=0.,
                 fillnan=True,
                 fillnan_robustness_factor=0.,
                 min_category_count=0,
                 copy=True):

        # param for objective variable
        if objective_col is not None:
            if type(objective_col) == str:
                self.objective_col = objective_col
            else:
                raise InvalidInputForSpecifyingObjectiveColumnError
        else:
            self.objective_col = None

        # params for scaling values
        self.objective_scaling = objective_scaling
        self.numerical_scaling = numerical_scaling
        self.scaling_robustness_factor = scaling_robustness_factor

        # params for filling missing values
        # If fillnan == False, missing categorical amd numerical variables will be -1 and nan, respectively.
        self.fillnan = fillnan
        self.fillnan_robustness_factor = fillnan_robustness_factor

        # param for category-threshold by minimum appearance of each category in each categorical variable
        self.min_category_count = min_category_count

        # param for internal copy.
        # set to False to perform inplace the input DataFrame and avoid a copy.
        self.copy = copy

        # internal flags
        self._fitted = False

    def fit_transform(self, df):
        if self._fitted:
            raise TransformerAlreadyFittedError

        if self.copy:
            df = df.copy()

        if logging:
            _start_message_fit_transform()

        if self.objective_col:
            y_is_numeric = pd.api.types.is_numeric_dtype(df[self.objective_col])
            y = df[self.objective_col].values.copy()
            if self.objective_scaling:
                if y_is_numeric:
                    self.y_mean, self.y_std = _mean_std_for_scaling(y, self.scaling_robustness_factor,
                                                                    self.objective_col)
                    y = (y - self.y_mean) / self.y_std
                else:
                    message = "Because the objective variable is categorical, " +\
                              "no scaling was performed to objective variable despite objective_scaling=True "
                    warnings.warn(message)
                    self.y_mean, self.y_std = None, None
            else:
                self.y_mean, self.y_std = None, None

        # information of variables
        self.variable_information = {
            "variables": None,
            "transform_index": None,
            "categorical_variables": [],
            "numerical_variables": [],
            "categorical_uniques": []
        }

        self.transforms = []
        categorical_transform_index = []
        numerical_transform_index = []
        for i, col in enumerate(df.columns):
            num_uniques = df[col].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            if (col == self.objective_col) or (num_uniques == 1):
                trans = Dropper()
                trans.fit_transform(col, self.objective_col)
                self.transforms.append(trans)

            elif (num_uniques > 2) and (not is_numeric):
                trans = Factorizer(self.min_category_count, self.fillnan)
                trans.fit_transform(df, col, self.variable_information)
                self.transforms.append(trans)
                categorical_transform_index.append(i)

            elif (num_uniques == 2) and (not is_numeric):
                trans = BinaryFactorizer(self.numerical_scaling, self.scaling_robustness_factor,
                                         self.fillnan, self.fillnan_robustness_factor)
                trans.fit_transform(df, col, self.variable_information)
                self.transforms.append(trans)
                numerical_transform_index.append(i)

            elif is_numeric:
                trans = NumericalHandler(self.numerical_scaling, self.scaling_robustness_factor,
                                         self.fillnan, self.fillnan_robustness_factor)
                trans.fit_transform(df, col, self.variable_information)
                self.transforms.append(trans)
                numerical_transform_index.append(i)

            else:
                message = "something wrong with column: " + col
                raise Exception(message)

        self.variable_information["variables"] = self.variable_information["categorical_variables"]\
                                                 + self.variable_information["numerical_variables"]
        self.variable_information["transform_index"] = categorical_transform_index + numerical_transform_index

        self.num_categoricals = len(self.variable_information["categorical_variables"])
        self.num_numericals = len(self.variable_information["numerical_variables"])

        x = self._df_to_numpy(df)

        if logging:
            _end_message_fit_transform(self.variable_information)

        self._fitted = True

        return (x, y) if self.objective_col else x

    def transform(self, df):
        if not self._fitted:
            raise TransformerNotFittedError

        if self.copy:
            df = df.copy()

        if len(df.columns) != len(self.transforms):
            raise WrongDataFrameConstructionError

        if self.objective_col in df.columns:
            y_exist = True
            y = df[self.objective_col].values.copy()
            if self.objective_scaling:
                y = (y - self.y_mean) / self.y_std
        else:
            y_exist = False

        for i, col in enumerate(df.columns):
            self.transforms[i].transform(df, col)

        x = self._df_to_numpy(df)

        return (x, y) if y_exist else x

    def variables(self):
        return self.variable_information["variables"]

    def categoricals(self):
        return self.variable_information["categorical_variables"]

    def numericals(self):
        return self.variable_information["numerical_variables"]

    def name_to_index(self, colname):
        if colname not in self.variable_information["variables"]:
            raise VariableNotExistError(colname)
        return self.variable_information["variables"].index(colname)

    def index_to_name(self, index):
        return self.variable_information["variables"][index]

    def is_numerical(self, index_or_colname):
        trans = self._get_transform(index_or_colname)
        if type(trans) == Factorizer:
            return False
        else:
            return True

    def categories(self, index_or_colname):
        trans = self._get_transform(index_or_colname)
        if type(trans) in [Factorizer, BinaryFactorizer]:
            return trans.categories
        else:
            raise HasNoDictionaryError

    def category_to_factorized(self, index_or_colname, category_name):
        trans = self._get_transform(index_or_colname)
        categories = self.categories(index_or_colname)
        if category_name not in categories:
            raise CategoryNotExistError(category_name)
        if type(trans) == Factorizer:
            return float(np.where(categories == category_name)[0][0])
        elif type(trans) == BinaryFactorizer:
            categories = self.categories(index_or_colname)
            if self.numerical_scaling:
                return float((np.where(categories == category_name)[0][0] - trans.mean) / trans.std)
            else:
                return float(np.where(categories == category_name)[0][0])

    def factorized_to_category(self, index_or_colname, factorized_value):
        trans = self._get_transform(index_or_colname)
        categories = self.categories(index_or_colname)

        if type(trans) == Factorizer:
            return _factorized_to_category(factorized_value, factorized_value, categories)

        elif type(trans) == BinaryFactorizer:
            if self.numerical_scaling:
                fixed_factorized_value = float(factorized_value * trans.std + trans.mean)
                # if not integer, raise error
                if not float.is_integer(fixed_factorized_value):
                    raise FactorizedNotExistError(factorized_value)
                return _factorized_to_category(fixed_factorized_value, factorized_value, categories)
            else:
                return _factorized_to_category(factorized_value, factorized_value, categories)

    def nuniques(self):
        return self.variable_information["categorical_uniques"]

    def nunique(self, index_or_colname=None):
        if index_or_colname is not None:
            trans = self._get_transform(index_or_colname)
            if type(trans) == Factorizer:
                return trans.num_uniques
            elif type(trans) == BinaryFactorizer:
                return 2
            elif type(trans) == NumericalHandler:
                raise WronglySpecifiedNumericalVariableError
        else:
            return self.variable_information["categorical_uniques"]

    def _df_to_numpy(self, df):
        x_categorical = df[self.variable_information["categorical_variables"]].values
        x_numerical = df[self.variable_information["numerical_variables"]].values
        return np.concatenate([x_categorical, x_numerical], axis=1)

    def _get_transform(self, index_or_colname):
        if type(index_or_colname) in [int, np.int, np.int8, np.int16, np.int32, np.int64]:
            return self.transforms[self.variable_information["transform_index"][index_or_colname]]
        elif type(index_or_colname) == str:
            if index_or_colname not in self.variable_information["variables"]:
                raise VariableNotExistError(index_or_colname)
            index = self.variable_information["variables"].index(index_or_colname)
            return self.transforms[self.variable_information["transform_index"][index]]
        else:
            raise InvalidInputForSpecifyingVariableError


############################
###  Internal Functions  ###
############################

def _start_message_fit_transform():
    print("Starting to fit a transformer of TransformDF2Numpy.")


def _end_message_fit_transform(info):
    print()
    print("Transformer fitted.")
    print("Number of the categorical variables:", len(info["categorical_variables"]))
    print("Number of the numerical variables:", len(info["numerical_variables"]))
    print("---------------------------------------------------")


def _message_variable_dropped(col_name):
    print("Variable Dropped because of containing only one unique value: (column: '%s')" % col_name)


def _message_categories_thresholed(col_name, num_valids, num_dropped):
    print("Categories thresholded: (column: '%s'), (valid categories: %d, dropped categories: %d)"
          % (col_name, num_valids, num_dropped))


def _message_numerical_nans_filled(col_name, nan_count, nan_value):
    print("Numerical NaNs filled with alternative value: (column: '%s'), (filled rows: %d, value: %f)"
          % (col_name, nan_count, nan_value))


def _message_categirical_nans_filled(col_name, nan_count, factorized_nan_value):
    message = "Categorical NaNs filled with alternative value: (column: '%s'), " % col_name +\
              "(filled rows: %d, factorized value: %f, category: '%s')" %\
              (nan_count, factorized_nan_value, NAN_CATEGORY)
    print(message)


def _factorized_to_category(fixed_factorized, factorized, categories):
    if fixed_factorized < len(categories):
        return categories[fixed_factorized]
    else:
        raise FactorizedNotExistError(factorized)


def _fit_factorize_fillnan_true(df, col_name):
    nan_count = df[col_name].isnull().sum()
    if nan_count:
        nan_value = NAN_CATEGORY         # nan will be replaced by new category
        df[col_name].fillna(nan_value, inplace=True)
        df[col_name], categories = df[col_name].factorize()
        factorized_nan_value = np.where(categories == NAN_CATEGORY)[0][0]
        if logging:
            _message_categirical_nans_filled(col_name, nan_count, factorized_nan_value)
    else:
        nan_value = df[col_name].mode()[0]      # future nan will be replaced by most frequently appeared category
        df[col_name], categories = df[col_name].factorize()
    return categories, nan_value


def _fit_factorize_fillnan_false(df, col_name):
    df[col_name], categories = df[col_name].factorize()
    return categories


def _numerical_nan_value(values, fillnan_robustness_factor):
    values = values[~np.isnan(values)]
    values = np.sort(values)
    start_index = int(len(values) / 2 * fillnan_robustness_factor)  # robustness_factorは片側
    gorl_index = int(len(values) - start_index)
    if start_index == gorl_index:
        gorl_index += 1
    nan_value = values[start_index:gorl_index].mean()
    return nan_value


def _mean_std_for_scaling(values, scaling_robustness_factor, col_name):
    values = values[~np.isnan(values)]
    values = np.sort(values)
    start_index = int(len(values) / 2 * scaling_robustness_factor)  # robustness_factorは片側
    gorl_index = int(len(values) - start_index)
    if start_index == gorl_index:
        gorl_index += 1
    std = values[start_index:gorl_index].std() + 0.000001
    if std == 0.000001:
        if logging:
            message = "Robust scaling of the variable:'%s' was failed due to infinite std appeared." % col_name\
                      + " The mean and std will be calculated by all values instead."
            warnings.warn(message)
        std = values.std() + 0.000001
        mean = values.mean()
        return mean, std
    else:
        mean = values[start_index:gorl_index].mean()
        return mean, std


##########################
###  Internal Classes  ###
##########################

class CategoryThreshold:
    def __init__(self):
        pass

    def fit_transform(self, df, col_name, min_count):
        val_cnt = df[col_name].value_counts()
        valid_categories_series = val_cnt >= min_count
        self.valid_categories = valid_categories_series[valid_categories_series].index

        drop_targets = list(set(df[col_name].values) - set(self.valid_categories) - set([np.nan]))
        df[col_name] = df[col_name].map(lambda x: DROPPED_CATEGORY if x in drop_targets else x)
        if len(drop_targets) != 0 and logging:
            _message_categories_thresholed(col_name, len(self.valid_categories), len(drop_targets))

    def transform(self, df, col_name):
        drop_targets = list(set(df[col_name].values) - set(self.valid_categories) - set([np.nan]))
        df[col_name] = df[col_name].map(lambda x: DROPPED_CATEGORY if x in drop_targets else x)


class Dropper:
    def __init__(self):
        pass

    def fit_transform(self, col_name, obj_col_name):
        self.col_name = col_name
        if logging and (col_name != obj_col_name):
            _message_variable_dropped(col_name)

    def transform(self, df, col_name):
        if col_name != self.col_name:
            raise WrongDataFrameConstructionError


class Factorizer:
    def __init__(self, min_category_count, fillnan_flag):
        self.min_category_count = min_category_count
        self.fillnan_flag = fillnan_flag

    def fit_transform(self, df, col_name, variable_info):
        self.col_name = col_name

        self.ct = CategoryThreshold()
        self.ct.fit_transform(df, col_name, min_count=self.min_category_count)

        if self.fillnan_flag:
            self.categories, self.nan_value = _fit_factorize_fillnan_true(df, col_name)
        else:
            self.categories = _fit_factorize_fillnan_false(df, col_name)

        variable_info["categorical_variables"].append(col_name)
        self.num_uniques = len(self.categories)
        variable_info["categorical_uniques"].append(self.num_uniques)

        # starting to create params used for an external one-hot-encoding function
        category_counts = df[col_name].value_counts()
        if -1 in category_counts.index.values:
            category_counts.drop(-1, axis=0, inplace=True)
        category_counts = category_counts.sort_index().values

        # means of one-hot-vectors
        self.categories_one_hot_means = category_counts / category_counts.sum()

        # standard deviations of one-hot-vectors
        self.categories_one_hot_stds = np.sqrt(
            self.categories_one_hot_means * (1 - self.categories_one_hot_means) ** 2 +
            (1 - self.categories_one_hot_means) * self.categories_one_hot_means ** 2
        )

    def transform(self, df, col_name):
        if col_name != self.col_name:
            raise WrongDataFrameConstructionError
        
        self.ct.transform(df, col_name)
        if self.fillnan_flag:
            df[col_name].fillna(self.nan_value, inplace=True)

        df[col_name] = self.categories.get_indexer(df[col_name])


class BinaryFactorizer:
    def __init__(self, scaling_flag, scaling_robustness_factor,
                 fillnan_flag, fillnan_robustness_factor):
        self.scaling_flag = scaling_flag
        self.scaling_robustness_factor = scaling_robustness_factor
        self.fillnan_flag = fillnan_flag
        self.fillnan_robustness_factor = fillnan_robustness_factor

    def fit_transform(self, df, col_name, variable_info):
        self.col_name = col_name

        df[col_name], self.categories = df[col_name].factorize()

        # fill nan
        nan_count = (df[col_name].values == -1).sum()
        if self.fillnan_flag and nan_count:
            df.loc[df[col_name] == -1, col_name] = np.nan
            self.nan_value = _numerical_nan_value(df[col_name].values, self.fillnan_robustness_factor)
            df[col_name].fillna(self.nan_value, inplace=True)
            if logging:
                _message_numerical_nans_filled(col_name, nan_count, self.nan_value)
        elif not self.fillnan_flag and nan_count:
            df.loc[df[col_name] == -1, col_name] = np.nan

        # scaling
        if self.scaling_flag:
            self.mean, self.std = _mean_std_for_scaling(df[col_name].values,
                                                        self.scaling_robustness_factor,
                                                        col_name)
            df[col_name] = (df[col_name].values - self.mean) / self.std

        variable_info["numerical_variables"].append(col_name)

    def transform(self, df, col_name):
        if col_name != self.col_name:
            raise WrongDataFrameConstructionError

        df[col_name] = self.categories.get_indexer(df[col_name])
        if self.fillnan_flag and (-1 in df[col_name].values):
            df.loc[df[col_name] == -1, col_name] = self.nan_value
        elif not self.fillnan_flag and (-1 in df[col_name].values):
            df.loc[df[col_name] == -1, col_name] = np.nan

        if self.scaling_flag:
            df[col_name] = (df[col_name].values - self.mean) / self.std


class NumericalHandler:
    def __init__(self, scaling_flag, scaling_robustness_factor,
                 fillnan_flag, fillnan_robustness_factor):
        self.scaling_flag = scaling_flag
        self.scaling_robustness_factor = scaling_robustness_factor
        self.fillnan_flag = fillnan_flag
        self.fillnan_robustness_factor = fillnan_robustness_factor

    def fit_transform(self, df, col_name, variable_info):
        self.col_name = col_name

        if self.fillnan_flag:
            self.nan_value = _numerical_nan_value(df[col_name].values, self.fillnan_robustness_factor)
            nan_count = (df[col_name].isnull()).sum()
            if nan_count:
                _message_numerical_nans_filled(col_name, nan_count, self.nan_value) if logging else None
                df[col_name].fillna(self.nan_value, inplace=True)

        if self.scaling_flag:
            self.mean, self.std = _mean_std_for_scaling(df[col_name].values, self.scaling_robustness_factor, col_name)
            df[col_name] = (df[col_name].values - self.mean) / self.std

        variable_info["numerical_variables"].append(col_name)

    def transform(self, df, col_name):
        if col_name != self.col_name:
            raise WrongDataFrameConstructionError

        if self.fillnan_flag:
            df[col_name].fillna(self.nan_value, inplace=True)

        if self.scaling_flag:
            df[col_name] = (df[col_name].values - self.mean) / self.std




