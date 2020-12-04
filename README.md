# double_debias
![Basic Build](https://github.com/joe5saia/double_debias/workflows/Double_Debias/badge.svg)


This package implements the double debiased estimator from ["Double/Debiased Machine Learning for Treatment and Structural Parameters"](https://economics.mit.edu/files/12538)
by Chernozhukov et. al. 

# installation 
`pip install double_debias_joe5saia`

# Usage
This package estimates models of the form y = theta D + g(z) + e where z is a high dimensional object. 

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
dd = double_debias(y=np.array([i for i in range(0,10)]), 
                   D= np.array([i//2 for i in range(0,10)]),
                   p.array([[i**2 for i in range(0,10)], [i**3 for i in range(0,10)]]).transpose(),
                   y_method= GradientBoostingRegressor(n_estimators=1000),
                   D_method= LinearRegression(),
                   n_folds=3)
dd.est_theta()
```

The user initializes the estimator object with the data for y, D, and z along with the method for estimating y ~ g(z) + e and D ~ f(z) + e. 
The `y_method` and `D_method` can be any model from the sklearn library that implements the fit and predict methods. The user may also supply their 
own class that implements these methods. This class does no parameter tuning or cross validation. Parameter tuning is left up to the user. 


# Custom Estimator Methods
The user may supply their own estimators if these are not available in sklearn. This module assumes that the class passed has the fit and predict methods, i.e. the following 
code must work
```python
z = np.array([[i**2 for i in range(0,10)], [i**3 for i in range(0,10)]]).transpose()
y = np.array([i for i in range(0,10)])
m = my_estimator()
m.fit(z, y)
m.predict(z)
```
