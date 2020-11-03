***
# df2numpy
A python package: **df2numpy**, containing a class ```TransformDF2Numpy``` and a function ```one_hot_encode```, 
is a simple and flexible tool for quick transformation from **pandas.DataFrame** to **numpy.array** dataset.  
  
*TransformDF2Numpy* is a class for fitting a transformer instance by a training pandas.DataFrame set.
This contains some utilities such as **re-transformation of new data**,
minimal **pre-processing**, and **access to variable information**.  
  
*one_hot_encolde* is a function that takes a instance of TransformDF2Numpy and its output numpy.array.
This returns one-hot-encoded numpy.array and the list of variable names.

## Overview

- *Quick fitting of a transformer*
  - Transform a training-set of pandas.DataFrame to the numpy.array dataset, and fit the transformer instance.
    The numpy.array contains the **factorized categorical variables (first half)**
    and the **numerical variables (second half)**.　　
　
 
- *Utilities of a fitted transformer instance.*
    - Transforming New DataFrame samely as DataFrame used for fitting.
        - This helps avoiding data leakage when evaluating a machine learning model, and also helps incorporating the model into a system.
    - Access to the variables information.
        - linking variable indices and the variable names
        - variable names (all, categorical, numerical)
        - linking factorized values and the category names
        - unique categories of categorical variables　　
　
  
- *Minimal pre-processing (optional).*
    - Scaling numerical variables.
        - robustness control by a parameter
    - Thresholding categorical variables by minimum count of each variable.
    - Filling missing values.
        - new category (or the most frequent category) for categorical variables.
        - mean value for numerical variables
        - robustness control by a parameter　　

- *Optional one-hot-encoding by a function* ```one_hot_encode```

> (*) Note that a categorical variable which has only two unique categories is treated as a numerical variable.

> (*) **Factorization**: The process of converting each element of a categorical variable into a corresponding positive index.

## Getting Started
```
pip install git+https://github.com/kitayama1234/TransformDF2Numpy
```

## Basic Usage
See demo.ipynb

## Advanced Usage
A DNN solution with a technique "Entity Enbedding" [1] for a tabular dataset: https://github.com/kitayama1234/Pytorch-Entity-Embedding-DNN-Regressor

> [1] Guo, Cheng, and Felix Berkhahn. "Entity embeddings of categorical variables." arXiv preprint arXiv:1604.06737 (2016).

***
# TransformDF2Numpy Documentation

## Parameters

####  ```objective_col```
- *str (optional, default None)*  
- The column name of objective variable.
If you specify this, the instance automatically find the column and the output numpy array will be splitted into x (explanatory variables) and y (objective variables).
  
#### ```objective_scaling```
- *bool (optional, default False)* 
- The flag for scaling objective variable.
  
#### ```numerical_scaling```
- *bool (optional, default False)* 
- The flag for scaling numerical variables.
  
#### ```scaling_robustness_factor```
- *float in range of [0. 1.] (optional, default 0.)*  
- The parameter to control robustness of scaling operation. Specifying a larger value will make it more robust against outliers.
   
#### ```fillnan```
- *bool (optional, default True)*  
- The flag to fill missing values (nan, NaN). If True, the numerical nan will be filled with the mean,
and the categorical nan will be filled as new category (or most frequent category).
If False, the numerical nan will not be filled,
and the categorical nan will be filled with -1.
  
#### ```fillnan_robustness_factor```
- *float in range of [0. 1.] (optional, default 0.)*  
- The parameter to control robustness of calculating the filling value to nan.
Specifying a larger value will make it more robust against outliers.
  
#### ```min_category_count```
- *integer (optional, default 0)*  
- The minimum number of appearance of each category, in each categorical variable.
A category with a number of occurrences less than this parameter will be thresholded,and treated as a new single category.
  
#### ```copy```
- *bool (optional, default True)*  
- Set to False to perform inplace the input DataFrame and avoid a copy.
  
## Methods

#### ```fit_transform(df)```
- *Inputs*: ```df```
  - training set of DataFrame  

- *Returns*: ```x```, (```y```)
  - ```x``` : The numpy.array containing factorized categorical variables (first half)
  and numerical variables (second half).
  The variables which have only two unique categories are treated as numerical variables.  
  - ```y``` : numpy array of objective variable (returned only when objective column exists)
  - The transformer instance will be fitted by the df.

#### ```transform(df)```
- *Inputs*: ```df```
  - testing set of DataFrame  

- *Returns*:   ```x```, (```y```)
  - ```x``` : numpy array of explanatory variables same as fit_transform()  
  - ```y``` : numpy array of objective variable (only when objective column exists)

#### ```variables()```
- *Returns*:  
  -the list of the name of all variables in order of the output numpy array
    
#### ```categoricals()```
- *Returns*:  
  - the list of the name of categorical variables in order of the output numpy array

#### ```numericals()```
- *Returns*:  
  - the list of the name of numerical variables in order of the output numpy array

#### ```name_to_index(colname)```
- *Inputs*: ```colname```  
  - column name of DataFrame  

- *Returns*:  
  - the corresponding column index of numpy array

#### ```index_to_name(index)```
- *Inputs*: ```index```  
  - column index of numpy array  

- *Returns*:  
  - the corresponding column name of DataFrame

#### ```is_numerical(index_or_colname)```
- *Inputs*: ```index_or_colname```
  - column column name of DataFrame, or column index of numpy array  

- *Returns*:
  - the bool indicating whether the variable is treated as a numerical variable or not

#### ```categories(index_or_colname)```
- *Inputs*: ```index_or_colname```  
  - column name of DataFrame, or column index of numpy array  

- *Returns*:  
  - the list of unique categories in the variable which index correspond to the factorized values

#### ```category_to_factorized(index_or_colname, category_name)```
- *Inputs*: ```index_or_colname```, ```category_name```  
  - ```index_or_colname``` : column name of DataFrame, or column index of numpy array  
  - ```category_name``` : name of the single category  

- *Returns*:  
  - the factorized value

#### ```factorized_to_category(index_or_colname, factorized_value)```
- *Inputs*: ```index_or_colname```, ```factorized_value```  
  - ```index_or_colname``` : column name of DataFrame, or column index of numpy array  
  - ```factorized_value``` : factorized value of the single category  

- *Returns*:  
  - the name of the single category

#### ```nuniques()```
- *Returns*:  
  - the list of the number of unique categories of the categorical variables

#### ```nunique(index_or_colname)```
- *Inputs*: ```index_or_colname```  
  - column name of DataFrame, or column index of numpy array  

- *Returns*:  
  - the number of unique categories of the categorical variable


## Attributes

#### ```y_mean```
  - the mean of the objective variable before scaling  
    
#### ```y_std```
  - the standard deviation of the objective variable before scaling  
    
#### ```num_categoricals```
  - the number of the categorical variables  
    
#### ```num_numericals```
  - the number of the numerical variables  

***

# Author
Masaki Kitayama  
email: kitayama-masaki@ed.tmu.ac.jp
