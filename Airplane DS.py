import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

data = pd.read_csv("AirPassengers.csv")

data.head()
data.shape
data.describe()
data.info()
data.duplicated().sum()

data["Month"] = pd.to_datetime(data["Month"], infer_datetime_format=True)

data = data.set_index("Month", inplace=False)

data.head()
data.info()

data.plot()

plt.plot(data, color="blue", label="Original")

mean_log = data.rolling(window=12).mean()  # yearly average
std_log = data.rolling(window=12).std()  # yearly diviation

plt.plot(mean_log, color="red", label="Rolling Mean")

plt.plot(data, color="blue", label="Original")
plt.plot(mean_log, color="red", label="Rolling Mean")
plt.plot(std_log, color="black", label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean & Standard Deviation (Logarithmic Scale)")

result = adfuller(data["#Passengers"])
print(result[1])

# logrithmic comutation to make the time series stationary
first_log = np.log(data)
first_log = first_log.dropna()
first_log.plot()

result = adfuller(first_log["#Passengers"])
print(result[1])

mean_log = first_log.rolling(window=12).mean()
std_log = first_log.rolling(window=12).std()

plt.plot(first_log, color="blue", label="Original")
plt.plot(mean_log, color="red", label="Rolling Mean")
plt.plot(std_log, color="black", label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean & Standard Deviation (Logarithmic Scale)")

mean_log.head(12)

new_data = first_log - mean_log
new_data = new_data.dropna()  # precaution if any null value
new_data.head()

# adfuller test for stationarity
result = adfuller(new_data["#Passengers"])
print(result[1])

mean_log = new_data.rolling(window=12).mean()
std_log = new_data.rolling(window=12).std()

plt.plot(new_data, color="blue", label="Original")
plt.plot(mean_log, color="red", label="Rolling Mean")
plt.plot(std_log, color="black", label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean & Standard Deviation (Logarithmic Scale)")

result = adfuller(new_data["#Passengers"])
print(result[1])

# seasonal Decompose
old_decompose_result = seasonal_decompose(data["#Passengers"].dropna())
old_decompose_result.plot()

# seasonal Decompose
decompose_result = seasonal_decompose(new_data["#Passengers"].dropna())
decompose_result.plot()

# ACF and PACF plots
acf_plot = acf(new_data)
pacf_plot = pacf(new_data)
plot_acf(acf_plot)
plot_pacf(pacf_plot)

# ARIMA model fitting
train = new_data.iloc[:120]["#Passengers"]
test = new_data.iloc[121:]["#Passengers"]

model = auto_arima(
    train,
    start_p=0,
    max_p=5,
    start_q=0,
    max_q=5,
    d=None,  # Let it determine the best differencing
    seasonal=False,  # Set to True and provide `m` if seasonal
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)

print(model.summary())

from statsmodels.tsa.arima_model import ARIMA

train = new_data.iloc[:120]["#Passengers"]
test = new_data.iloc[121:]["#Passengers"]

model = sm.tsa.arima.ARIMA(train, order=(3, 0, 2))
model_fit = model.fit()
model_fit.summary()

from statsmodels.tsa.arima_model import ARIMA

train = new_data.iloc[:120]["#Passengers"]
test = new_data.iloc[121:]["#Passengers"]

model = sm.tsa.arima.ARIMA(train, order=(3, 0, 2))
model_fit = model.fit()
model_fit.summary()

new_data["predict"] = model_fit.predict(
    start=len(train), end=len(train) + len(test) - 1, dynamic=True
)
new_data[["#Passengers", "predict"]].plot()

from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

model = SARIMAX(train, order=(3, 0, 2), seasonal_order=(3, 0, 2, 6))
model = model.fit()

new_data["predict"] = model.predict(
    start=len(train), end=len(train) + len(test) - 1, dynamic=True
)
new_data[["#Passengers", "predict"]].plot()

# predicting the projections for the next 5 years
forecast = model.forecast(steps=60)
new_data.plot()
forecast.plot()