from unittest import TestCase
from math import isclose
import numpy as np
import double_debias as ddd
from sklearn.ensemble import GradientBoostingRegressor


ob = dd.double_debias(y=np.array([i for i in range(0,10)]), 
                   D= np.array([i//2 for i in range(0,10)]),
                   z=np.array([[i**2 for i in range(0,10)], [i**3 for i in range(0,10)]]).transpose(),
                   y_method= GradientBoostingRegressor(n_estimators=1000),
                   D_method= GradientBoostingRegressor(n_estimators=1000),
                   n_folds=3)


ob.est_theta()

class TestDD(TestCase):
    def test_run(self):
        ob = dd.double_debias(y=np.array([i for i in range(0,10)]), 
                   D= np.array([i//2 for i in range(0,10)]),
                   z=np.array([[i**2 for i in range(0,10)], [i**3 for i in range(0,10)]]).transpose(),
                   y_method= GradientBoostingRegressor(n_estimators=1000),
                   D_method= GradientBoostingRegressor(n_estimators=1000),
                   n_folds=3)
        theta = ob.est_theta()           
        self.assertTrue(isclose(theta, -0.6, abs_tol=1e-1) )