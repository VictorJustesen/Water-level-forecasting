import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import xgboost as xgb

import gpflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


#---------------------------------------------------------------
#---------------------------------------------------------------
def moving_average(data, window_size, n_forecast):
    return data.rolling(window = window_size).mean().iloc[-n_forecast:]

#---------------------------------------------------------------
def autoregression(data, start, end):
    res = AutoReg(data, lags = 1).fit()
    return res.model.predict(res.params, start = start, end = end)

#---------------------------------------------------------------
def simple_exponential_smoothing(data, n_forecast):
    model = SimpleExpSmoothing(data)
    model_fit = model.fit()
    return model_fit.forecast(n_forecast)

#---------------------------------------------------------------
def holtWinter_smoothing(data, n_forecast):
    model = ExponentialSmoothing(data)
    model_fit = model.fit()
    return model_fit.forecast(n_forecast)

#---------------------------------------------------------------
def arima_model(data, n_forecast):
    model = ARIMA(endog = data)
    model_fit = model.fit()
    return model_fit.forecast(steps = n_forecast)

#---------------------------------------------------------------
def linear_regression_model(Xtrain, ytrain, Xtest):
    
    model = LinearRegression()
    model.fit(Xtrain, ytrain)
    return model.predict(Xtest)

#---------------------------------------------------------------
def GPR(Xtrain, ytrain, Xtest):

  

    kernel = gpflow.kernels.RBF(lengthscales = 10)

    model = gpflow.models.SGPR(data = (Xtrain, ytrain), 
                               kernel = kernel, 
                               inducing_variable = Xtrain[::10])

    #opt = gpflow.optimizers.Scipy()
    opt = tf.optimizers.Adam()
    opt_logs = opt.minimize(model.training_loss, 
                            model.trainable_variables, 
                            options = dict(maxiter = 100))
    predictions, _ = model.predict_y(Xtest)
    return predictions

#---------------------------------------------------------------

#---------------------------------------------------------------
# def XGB(Xtrain, ytrain, Xtest):

#    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
#                            n_estimators=1000,
#                            early_stopping_rounds=50,
#                            objective='reg:linear',
#                            max_depth=3,
#                            learning_rate=0.01)
#     reg.fit(Xtrain, Ytrain,
#         eval_set=[(Xtrain, Ytrain), (Xtest, Ytest)],
#         verbose=0)        
#     return  reg.predict(Xtest)

#---------------------------------------------------------------

