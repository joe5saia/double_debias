import pytest
import numpy as np
from double_debias import double_debias
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


def basetest(nregressors):
    # Linear regression with a bunch of meaningless regressors. Y~D + epsilon in the base model
    X, y, w = make_regression(n_samples=10000, n_informative=nregressors, n_features=10, coef=True, random_state=1, bias=3.5, shuffle=False, noise=0.001)
    dd = double_debias(y=y, D=X[:, :nregressors], z=X[:, nregressors:], y_method=LinearRegression(), D_method=LinearRegression(), n_folds=3)
    theta = dd.est_theta()
    np.testing.assert_allclose(w[:nregressors], theta, rtol=0.1)


def test_1d():
    # Test for 2-D D matrix with a single column
    basetest(1)


def test_2d():
    # Test for 2-D D matrix with 2 columns
    basetest(2)


def test_confunders():
    # Linear regression with confunders. Y ~ D + z + epsilon and E[DZ] != 0
    # Verify that this works when there are confunders
    z, D, w = make_regression(n_samples=100000, n_informative=8, n_features=10, coef=True, random_state=1, bias=3.5, shuffle=False, noise=10)
    y = D + z.dot(np.random.rand(z.shape[1]))
    D = D.reshape(-1, 1)
    dd = double_debias(y=y, D=D, z=z, y_method=LinearRegression(), D_method=LinearRegression(), n_folds=3)
    theta = dd.est_theta()
    np.testing.assert_allclose(1, theta, rtol=0.1)


def test_nonlinear(nregressors=2):
    # Nonlinear regression with a bunch of meaningless regressors. Y ~ D + epsilon in the base model
    # Tests that this works for 2-D D matrix with a single column
    X, y, w = make_regression(n_samples=3000, n_informative=nregressors, n_features=10, coef=True, random_state=1, bias=3.5, shuffle=False, noise=0.001)
    dd = double_debias(y=y, D=X[:, :nregressors], z=X[:, nregressors:], y_method=RandomForestRegressor(), D_method=RandomForestRegressor(), n_folds=3)
    theta = dd.est_theta()
    np.testing.assert_allclose(w[:nregressors], theta, rtol=0.1)


def test_nonlinear2():
    # Nonlinear regression with a bunch of meaningless regressors. Y ~ D + z+ epsilon in the base model
    # Tests that this works for 2-D D matrix with 1 column
    z, D, w = make_regression(n_samples=100000, n_informative=2, n_features=10, coef=True, random_state=1, bias=3.5, shuffle=False, noise=10)
    y = D + (z[:, :2]**2).dot(np.random.rand(2))
    D = D.reshape(-1, 1)
    dd = double_debias(y=y, D=D, z=z, y_method=RandomForestRegressor(), D_method=RandomForestRegressor(), n_folds=3)
    theta = dd.est_theta()
    np.testing.assert_allclose(1, theta, rtol=0.1)


def test_nonlinear3():
    # Nonlinear regression with a bunch of meaningless regressors. Y ~ D + z+ epsilon in the base model
    # Tests that this works for 2-D D matrix with 2 columns
    # Non optimized model so results are only approximently correct.
    z, D1, w = make_regression(n_samples=100000, n_informative=2, n_features=10, coef=True, random_state=1, bias=3.5, shuffle=False, noise=30)
    z, D2, w = make_regression(n_samples=100000, n_informative=2, n_features=10, coef=True, random_state=5, bias=3.5, shuffle=False, noise=30)
    D = np.stack([D1, D2], axis=1)
    y = D.dot(np.array([-20.0, 40.0])) + (z[:, :2]**2).dot(np.random.rand(2))
    dd = double_debias(y=y, D=D, z=z, y_method=RandomForestRegressor(), D_method=RandomForestRegressor(), n_folds=3)
    theta = dd.est_theta()
    np.testing.assert_allclose([-20.0, 40.0], theta, atol=5)
