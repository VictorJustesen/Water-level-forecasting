import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from data import loadDataset
from models import *

#---------------------------------------------------------------
#---------------------------------------------------------------
def run():
    '''
    Run selected algorithms on the dataset (set in LoadDataset)

    In the regression task y = f(x),
    x is time
    y is the water level
    This is a very simplistic approach, need to add more variables 
    like precipitation, temperature etc.
    '''
    x, ts = loadDataset()

    # Split 80/20 train/test
    n = ts.shape[0]
    trainData = ts[0:int(0.8 * n)]
    ntrain = trainData.shape[0]
    testData = ts[int(0.8 * n):]
    ntest = testData.shape[0]
    Xtrain = np.array(range(0, ntrain))
    Xtest = np.array(range(0, ntest))
    r2s = []

    models = ["mavg", "areg", "simple", "exp", "arima", "linreg"]

    for model in models:

        if model == "mavg":
            forecast = moving_average(trainData, 10, ntest)
        if model == "areg":
            forecast = autoregression(trainData, 
                                      start = int(0.8 * n), 
                                      end = n - 1)
        if model == "simple":
            forecast = simple_exponential_smoothing(trainData, ntest)

        if model == "exp":
            forecast = holtWinter_smoothing(trainData, ntest)

        if model == "arima":
            arima_model(trainData, n_forecast = ntest)
        if model == "linreg":
            forecast = linear_regression_model(Xtrain = Xtrain,
                                               ytrain = trainData.values,
                                               Xtest = Xtest)
        if model == "gpr":
            forecast = GPR(Xtrain = Xtrain,
                           ytrain = trainData.values,
                           Xtest = Xtest)

        print("Model = ", model)
        r2 = r2_score(testData, forecast)
        print("R2 = ", r2)
        r2s.append(r2)

        # Plot original time series data and forecasted values
        plt.figure(figsize = (10, 4))
        plt.plot(ts, label = 'Original Data')
        plt.plot(range(int(0.8 * n), n), 
                forecast, 
                color = 'red', 
                label = 'Forecast')
        plt.title('Forecast ({})'.format(model))
        plt.xlabel('Time')
        plt.ylabel('Water level')
        plt.legend()
        plt.savefig("forecast_{}.png".format(model))
        plt.close()
    
    results_df = pd.DataFrame(data = models, columns =["Model"])
    results_df["R2"] = r2s
    results_df.to_excel("forecasting_results.xlsx", index = False)

if __name__ == "__main__":
    run()