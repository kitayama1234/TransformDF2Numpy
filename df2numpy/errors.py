class InvalidInputForSpecifyingObjectiveColumnError(ValueError):
    def __init__(self) -> None:
        super(ValueError, self).__init__("objective_col must be specified as str or None")


class InvalidInputForSpecifyingVariableError(ValueError):
    def __init__(self) -> None:
        super(ValueError, self).__init__("Input must be an index (int) or a name of variable (str)")


class HasNoDictionaryError(ValueError):
    def __init__(self) -> None:
        super(ValueError, self).__init__("Specified variable has no dictionary.")


class CategoryNotExistError(ValueError):
    def __init__(self, category_name) -> None:
        message = "The category named '%s' does not exist in the dictionary of the variable" % category_name
        super(ValueError, self).__init__(message)


class FactorizedNotExistError(ValueError):
    def __init__(self, factorized_value) -> None:
        message = "The factorized value %f does not correspond to any category." % factorized_value
        super(ValueError, self).__init__(message)


class WronglySpecifiedNumericalVariableError(ValueError):
    def __init__(self) -> None:
        super(ValueError, self).__init__("Specified variable is a numerical variable.")


class VariableNotExistError(ValueError):
    def __init__(self, colname) -> None:
        message = "The variable named '%s' does not exist in the list of the variables" % colname
        super(ValueError, self).__init__(message)


class WrongDataFrameConstructionError(ValueError):
    def __init__(self) -> None:
        super(ValueError, self).__init__("Could not transform. DataFrame construction is wrong.")


class InvalidTransformerError(ValueError):
    def __init__(self) -> None:
        super(ValueError, self).__init__("The first argument must be a instance of the TransformDF2Numpy class.")


class NoCategoricalVariableError(ValueError):
    def __init__(self) -> None:
        message = "According to the information of the transformer, the data contains no categorical variable."
        super(ValueError, self).__init__(message)


class InvalidNumpyArrayShapeError(ValueError):
    def __init__(self) -> None:
        super(ValueError, self).__init__("The shape of input numpy array must be (rows, columns).")


class InvalidNumpyArrayColumnsError(ValueError):
    def __init__(self) -> None:
        message = "The number of columns in the input numpy array does not match "\
                  + "the number of variables from the transformer's information."
        super(ValueError, self).__init__(message)



