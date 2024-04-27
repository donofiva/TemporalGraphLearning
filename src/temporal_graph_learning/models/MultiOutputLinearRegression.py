from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


class MultiOutputLinearRegression(MultiOutputRegressor):

    def __init__(self, *, fit_intercept: bool = True, copy_X: bool = True, n_jobs: int = None):
        super().__init__(
            estimator=LinearRegression(
                fit_intercept=fit_intercept,
                copy_X=copy_X,
                n_jobs=n_jobs
            )
        )
