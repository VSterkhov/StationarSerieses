import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

################
 # Dickey-Fuller
##################
def test_stationarity(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for [key, value] in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def tsplot(y, lags=None, figsize=(14, 8), style='bmh'):
    test_stationarity(y)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        plt.figure(figsize=figsize)
        layout = (5, 1)
        ts_ax = plt.subplot2grid(layout, (0, 0), rowspan=2)
        acf_ax = plt.subplot2grid(layout, (2, 0))
        pacf_ax = plt.subplot2grid(layout, (3, 0))
        qq_ax = plt.subplot2grid(layout, (4, 0))

        y.plot(ax=ts_ax, color='blue', label='Or')
        ts_ax.set_title('Original')

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        
        plt.tight_layout()
    return


sales_of_company_x = pd.read_csv("./monthly-sales-of-company-x-jan-6.csv")
robberies_in_boston = pd.read_csv("./monthly-boston-armed-robberies-j.csv")
airlines_passengers = pd.read_csv("./international-airline-passengers.csv")
mean_monthly_temp = pd.read_csv("./mean-monthly-air-temperature-deg.csv")
dowjones_closing = pd.read_csv("./weekly-closings-of-the-dowjones-.csv")
female_births = pd.read_csv("./daily-total-female-births-in-cal.csv")

all_series = {
    "sales": sales_of_company_x["Count"],
    "robbies": robberies_in_boston["Count"],
    "air_thousand": airlines_passengers["Count"],
    "air_temp": mean_monthly_temp["Deg"],
    "dow_jones": dowjones_closing["Close"],
    "calif": female_births["Count"]
}


with plt.style.context('bmh'):
    plt.figure(figsize=(16, 8))
    layout = (3, 2)
    for i, key in enumerate(all_series.keys()):
        x = i % 2
        y = int((i - x) / 2)
        
        ts_ax = plt.subplot2grid(layout, (y, x))
        all_series[key].plot(ax=ts_ax, color='blue')
        ts_ax.set_title(key)
        
    plt.tight_layout()


#plot_ts_and_points(sales_of_company_x['Count'], 2, 4)
#plot_ts_and_points(robberies_in_boston['Count'], 2, 4)
#plot_ts_and_points(airlines_passengers['Count'], 2, 4)
#plot_ts_and_points(mean_monthly_temp['Deg'], 2, 4)
#plot_ts_and_points(dowjones_closing['Close'], 2, 4)
#plot_ts_and_points(female_births['Count'], 2, 4)

''
mdl = smt.AutoReg(all_series["sales"],lags=30).fit()
print(mdl.params)
all_series["sales"] = np.diff(all_series["sales"])
all_series["sales"] = all_series["sales"][14:] - all_series["sales"][:-14]
tsplot(all_series["sales"], lags=12)

all_series["dow_jones"] = np.diff(all_series["dow_jones"])
tsplot(all_series["dow_jones"], lags=30)


mdl = smt.AutoReg(all_series["air_thousand"],lags=30).fit()
print(mdl.params)
all_series["air_thousand"] = np.diff(all_series["air_thousand"])
all_series["air_thousand"] = all_series["air_thousand"][12:] - all_series["air_thousand"][:-12]
tsplot(all_series["air_thousand"], lags=30)


all_series["dow_jones"] = np.diff(all_series["dow_jones"])
tsplot(all_series["dow_jones"], lags=30)
all_series["robbies"] = np.diff(all_series["robbies"])
tsplot(all_series["robbies"], lags=30)
