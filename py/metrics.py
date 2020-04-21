from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import iqr
from math import sqrt

def RMSE(obs, pred):
    return sqrt(mean_squared_error(obs, pred))

def RPIQ(obs, pred):
    return iqr(obs) / RMSE(obs, pred)

def error_metrics(obs, pred):
    return {"RMSE": RMSE(obs, pred), "R2": r2_score(obs, pred), "RPIQ": RPIQ(obs, pred)}