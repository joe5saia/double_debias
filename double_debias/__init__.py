import numpy as np
import copy
from sklearn.model_selection import KFold


class double_debias:
    """
    Estimator class for implementing Double Debiased Learning algorithm.

    ---

    The model estimated is y ~ theta D + g(z) + e
    After initializing object, double_debias.est_theta() estimates theta.
    """

    def __init__(self, y=None, D=None, z=None, D_method=None, y_method=None, n_folds=2):
        """
        Constructor for the base information for double_debias

        Arguments:
        ---
            y : 1-D array
                Dependent data
            D : 1-D array
                Treatment data
            z : 2-D array
                Confunders
            D_method : sklearn model
                slearn model such as sklearn.ensemble.GradientBoostingRegressor. Can be any class that implements the sklearn
                API of
                1) Initialize as D_method()
                2) D_method.fit()
                3) D_method.predict()
            y_method : sklearn model
                slearn model such as sklearn.ensemble.GradientBoostingRegressor. Can be any class that implements the sklearn
                API of
                1) Initialize as y_method()
                2) y_method.fit()
                3) y_method.predict()
            n_folds : Int
                Number of folds to be used in estimation. Needs to be at least 2
        """
        self.y = y
        self.D = D
        self.z = z
        self.nobs = y.shape[0]
        self.KFolds = KFold(n_splits=n_folds)
        self.methods = {'y': y_method, 'D': D_method}
        self.models = {'y': [copy.deepcopy(self.methods['y']) for i in range(self.KFolds.n_splits)],
                       'D': [copy.deepcopy(self.methods['D']) for i in range(self.KFolds.n_splits)]}

    @staticmethod
    def selector_check(selector):
        """" Validates that selector is a valid option, raises an AttributeError if not."""
        if selector not in ['y', 'D']:
            print(f"selector = {selector}. Selector must be either 'y' or 'D'")
            raise AttributeError

    @staticmethod
    def theta_formula(ytilde, V, D):
        """"Nyman orthogonal estimator for theta"""
        return np.mean(V*ytilde)/np.mean(V*D)

    def KFolds_split_(self, selector):
        """" Returns the indices for the KFolds object for y or D"""
        self.selector_check(selector)
        return self.KFolds.split(self.z, getattr(self, selector))

    def est_models_(self, selector):
        """ Estimate the y_models or D_models specified by selector"""
        self.selector_check(selector)
        for idx, (train, test) in enumerate(self.KFolds_split_(selector)):
            self.models[selector][idx].fit(
                self.z[train], getattr(self, selector)[train])

    def predict_(self, selector):
        """ Returns a generator for the predicted values for each fold of the y or D data specified by selector"""
        self.selector_check(selector)
        return (self.models[selector][idx].predict(self.z[test]) for idx, (train, test) in enumerate(self.KFolds_split_(selector)))

    def residualize_(self, selector):
        """ Returns a generator for the residual values for each fold of the y or D data specified by selector"""
        return (getattr(self, selector)[test] - self.models[selector][idx].predict(self.z[test]) for idx, (train, test) in enumerate(self.KFolds_split_(selector)))

    def est_thetas(self):
        """ Estimate theta for each fold of the data, store as self.theta and return the array """
        self.thetas = np.array([self.theta_formula(Y, V, D[1]) for (Y, V, D) in zip(
            self.residualize_('y'), self.residualize_('D'), self.KFolds_split_('D'))])
        return self.thetas

    def est_theta(self):
        """
        Runs the full estimation loop for theta

        ---
        Estimates the model for each fold of the y and D data and then estimates theta for each fold of the data.
        Returns the mean of the thetas.
        """
        self.est_models_('y')
        self.est_models_('D')
        self.est_thetas()
        return np.mean(self.thetas)
