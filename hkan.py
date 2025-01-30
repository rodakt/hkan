import itertools
import tqdm

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


MININTERVAL = 1
TQDM_DISABLE = True 


def set_mininterval(mininterval):
    global MININTERVAL
    MININTERVAL = mininterval


def set_tqdm_disable(disable):
    global TQDM_DISABLE
    TQDM_DISABLE = disable

class Sigmoid:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return 1 / (1 + np.exp(-self.s * x))

class Gaussian:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.exp(-((self.s * x) ** 2))

class ReLU:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.maximum(0, self.s * x)

class Tanh:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.tanh(self.s * x)

class Softplus:
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.log(1 + np.exp(self.s * x))
    
class Identity:
    def __call__(self, x):
        return x


def make_centers(n_vars_out, n_vars_in, n_basis, centers, X=None):
    """
    Make the centers for the basis functions.

    Parameters:
    - n_vars_out: Number of output variables.
    - n_vars_in: Number of input variables.
    - n_basis: Number of basis functions.
    - centers: Method to determine the centers of the basis functions.
    - X: Input data.

    Returns:
    - The centers of the basis functions. ndarray of shape (n_vars_out, n_vars_in, n_basis)
    """
    if centers == "random":
        return np.random.uniform(0, 1, (n_vars_out, n_vars_in, n_basis))
    elif centers == "equally_spaced":
        return np.tile(np.linspace(0, 1, n_basis), (n_vars_out, n_vars_in, 1))
    elif centers == "random_data_points":
        C = np.empty((n_vars_out, n_vars_in, n_basis))
        for q in range(n_vars_out):
            for p in range(n_vars_in):
                C[q, p, :] = np.random.choice(X[:, p], n_basis)
        return C
    else:
        raise ValueError(
            "Possible values for 'centers' are 'random', 'equally_spaced', or 'random_data_points'."
        )


def apply_basis_fn(X, centers_arr, basis_fn, q, p):
    """
    Apply the basis function to the difference between the input data and the centers.

    Parameters:
    - X: Input data. Column vector of shape (n_samples, n_vars_in).
    - centers_arr: Centers of the basis functions. ndarray of shape (n_vars_out, n_vars_in, n_basis).
    - basis_fn: Basis function to apply.
    - q: Index of the output variable.
    - p: Index of the input variable.

    Returns:
    - The result of applying the basis function to the difference between the input data and the centers.
    """
    n_samples, n_vars_in = X.shape
    n_vars_out, _, n_basis = centers_arr.shape
    return basis_fn(
        X[:, p].reshape(n_samples, 1) - centers_arr[q, p, :].reshape(1, n_basis)
    )


class ExpandingLayer(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        n_vars_out,
        n_basis=10,
        centers="random",
        basis_fn=Sigmoid(),
        base_regressor=None,
    ):
        """
        Initialize the ExpandingLayer.

        Parameters:
        - n_vars_out: Number of output variables.
        - n_basis: Number of basis functions.
        - centers: Method to determine the centers of the basis functions.
        - basis_fn: Basis function to use, default is sigmoid.
        - base_regressor: Base regressor to use, default is LinearRegression.
        """
        self.n_vars_out = n_vars_out
        self.n_basis = n_basis
        self.centers = centers
        self.basis_fn = basis_fn
        self.base_regressor = base_regressor

    def fit(self, X, y=None):
        """
        Fit the model using the input data X and target y.

        Parameters:
        - X: Input data.
        - y: Target data.

        Returns:
        - self: Fitted estimator.
        """

        if self.base_regressor is None:
            self.base_regressor = LinearRegression(fit_intercept=False)

        self.n_samples_, self.n_vars_in_ = X.shape

        self.centers_arr_ = make_centers(
            self.n_vars_out, self.n_vars_in_, self.n_basis, self.centers, X
        )
        assert self.centers_arr_.shape == (
            self.n_vars_out,
            self.n_vars_in_,
            self.n_basis,
        ), (
            f"Centers shape is {self.centers_arr_.shape}, "
            f"expected {(self.n_vars_out, self.n_vars_in_, self.n_basis)}"
        )

        self.models_ = []

        for q, p in tqdm.tqdm(
            itertools.product(range(self.n_vars_out), range(self.n_vars_in_)),
            desc="Fitting 1D regressors",
            total=self.n_vars_out * self.n_vars_in_,
            mininterval=MININTERVAL,
            disable=TQDM_DISABLE,
        ):
            transformed_features = apply_basis_fn(
                X, self.centers_arr_, self.basis_fn, q, p
            )
            reg = clone(self.base_regressor).fit(transformed_features, y)
            self.models_.append((q, p, reg))

        return self

    def transform(self, X):
        """
        Transform the input data X.

        Parameters:
        - X: Input data to transform.

        Returns:
        - Transformed data.
        """
        assert (
            self.n_vars_in_ == X.shape[1]
        ), f"Input data has {X.shape[1]} features but expected {self.n_vars_in_}"

        out = np.empty((self.n_vars_out, self.n_vars_in_, X.shape[0]))

        for q, p, reg in self.models_:
            transformed_features = apply_basis_fn(
                X, self.centers_arr_, self.basis_fn, q, p
            )
            out[q, p, :] = reg.predict(transformed_features)

        return out


class ConnectingLayer(TransformerMixin, RegressorMixin, BaseEstimator):

    def __init__(self, base_regressor=None):
        self.base_regressor = base_regressor

    def fit(self, X, y=None):
        if self.base_regressor is None:
            self.base_regressor = LinearRegression(fit_intercept=True)

        self.n_vars_out_, self.n_vars_in_, self.n_samples_ = X.shape
        self.models_ = []

        for q in tqdm.tqdm(
            range(self.n_vars_out_),
            desc="Fitting connecting regressors",
            total=self.n_vars_out_,
            mininterval=MININTERVAL,
            disable=TQDM_DISABLE,
        ):
            reg = clone(self.base_regressor).fit(X[q, :, :].T, y)
            self.models_.append((q, reg))

        return self

    def transform(self, X):
        assert (
            self.n_vars_out_ > 1
        ), "Unable to transform data with only one output variable. Use predict instead."

        out = np.empty((self.n_vars_out_, X.shape[2]))

        for q, reg in self.models_:
            out[q, :] = reg.predict(X[q, :, :].T)

        return out.T

    def predict(self, X):
        assert (
            self.n_vars_out_ == 1
        ), "Unable to predict data with more than one output variable. Use transform instead."

        _, reg = self.models_[0]
        return reg.predict(X[0, :, :].T)


def make_hkan_layer(
    *,
    layer_idx,
    n_vars_out,
    n_basis=10,
    centers="random",
    basis_fn=Sigmoid(),
    expanding_base_regressor=None,
    connecting_base_regressor=None,
):
    """Pipeline of ExpandingLayer and ConnectingLayer."""
    steps = [
        (
            f"expanding_layer_{layer_idx}",
            ExpandingLayer(
                n_vars_out=n_vars_out,
                n_basis=n_basis,
                centers=centers,
                basis_fn=basis_fn,
                base_regressor=expanding_base_regressor,
            ),
        ),
        (
            f"connecting_layer_{layer_idx}",
            ConnectingLayer(base_regressor=connecting_base_regressor),
        ),
    ]
    return Pipeline(steps)


def extend_hkan(
    model,
    *,
    layer_idx=None,
    n_vars_out=1,
    n_basis=10,
    centers="random",
    basis_fn=Sigmoid(),
    expanding_base_regressor=None,
    connecting_base_regressor=None,
):
    """
    Extend the HKAN model with additional HKAN layer.

    Parameters:
    - model: The HKAN model to extend.
    - n_vars_out: Number of output variables.
    - n_basis: Number of basis functions.
    - centers: Method to determine the centers of the basis functions.
    - basis_fn: Basis function to use, default is sigmoid.
    - expanding_base_regressor: Base regressor for the expanding layer.
    - connecting_base_regressor: Base regressor for the connecting layer.

    Returns:
    - The extended HKAN model.
    """
    if layer_idx is None:
        layer_idx = len(model.steps) // 2

    new_layer = make_hkan_layer(
        layer_idx=layer_idx,
        n_vars_out=n_vars_out,
        n_basis=n_basis,
        centers=centers,
        basis_fn=basis_fn,
        expanding_base_regressor=expanding_base_regressor,
        connecting_base_regressor=connecting_base_regressor,
    )

    return Pipeline(model.steps + new_layer.steps)
