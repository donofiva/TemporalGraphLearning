from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


class MultiTargetLinearRegression(MultiOutputRegressor):

    def __init__(self, *, fit_intercept: bool = True, copy_X: bool = True, n_jobs: int = None):
        super().__init__(
            estimator=LinearRegression(
                fit_intercept=fit_intercept,
                copy_X=copy_X,
                n_jobs=n_jobs
            )
        )

    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    @property
    def copy_X(self):
        return self.estimator.copy_X

    @copy_X.setter
    def copy_X(self, value):
        self.estimator.copy_X = value
