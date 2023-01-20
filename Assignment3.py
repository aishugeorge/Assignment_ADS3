#Importing required modules
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.optimize import curve_fit
import itertools as iter


indicator = ["EN.ATM.CO2E.PC","EN.ATM.METH.KT.CE"]
country_code = ['AUS','CHN','IND','FRA','KOR','NZL','DEU','USA','JPN']


#Function to read the data frameand returning the data frame
def read_data(indicator,country_code):
  df = wb.data.DataFrame(indicator, country_code, mrv=30)
  #Removing 'YR' and assigning new index name
  df.columns = [i.replace('YR','') for i in df.columns]
  df=df.stack().unstack(level=1)
  df.index.names = ['Country', 'Year']
  df = df.reset_index()
  return df


data = read_data(indicator, country_code)
print(data.columns)
data["Year"] = pd.to_numeric(data["Year"])


#Function to normalise data in the data frame
def norm_df(df):
  y = df.iloc[:,2:]
  df.iloc[:,2:] = (y-y.min())/ (y.max() - y.min())
  return df

#Normalising dataframe
dt_norm = norm_df(data)
df_fit = dt_norm.drop('Country', axis = 1)

# Performing K means clustering 
k = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(df_fit)
sns.scatterplot(data=dt_norm, x="Country", y="EN.ATM.CO2E.PC", hue=k.labels_)
plt.legend()
plt.show()


#Function for calculating error range
def err_ranges(x, func, param, sigma):
#initiating the arrays for lower and upper limit
  lower = func(x, *param)
  upper = lower
  uplow = []
#list to hold upper and lower limits for parameters
  for p,s in zip(param, sigma):
    pmin = p - s
    pmax = p + s
    uplow.append((pmin, pmax))
    pmix = list(iter.product(*uplow))
  for p in pmix:
    y = func(x, *p)
    lower = np.minimum(lower, y)
    upper = np.maximum(upper, y)
  return lower, upper


#Curve fit functionimplementation for New Zealand
data1 = data[(data['Country'] == 'NZL')]
print(data1)

dataval = data1.values
x, y = dataval[:, 1], dataval[:, 2]

def fun(x, a, b, c):
  return a*x**2+b*x+c

prmet, cov = opt.curve_fit(fun, x, y)
data1["pop_log"] = fun(x, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x, data1["pop_log"], label="Fit")
plt.plot(x, y, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in New Zealand")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
plt.savefig("Newzealand.png")

sigma = np.sqrt(np.diag(cov))
print(sigma)

low, up = err_ranges(x, fun, prmet, sigma)
print("Forcasted CO2 emission")
low, up = err_ranges(2030, fun, prmet, sigma)
print("2030 between", low, "and", up)

##Dataframe containing the data of country Korea
dt2 = data[(data['Country'] == 'KOR')]
print(dt2)

val2 = dt2.values
x2, y2 = val2[:, 1], val2[:, 2]

prmet, cov = opt.curve_fit(fun, x2, y2)
dt2["pop_log"] = fun(x2, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x2, dt2["pop_log"], label="Fit")
plt.plot(x2, y2, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 Emission')
plt.title("CO2 Emission rate in Korea")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
plt.savfig("Korea.png")

#Extracting the sigmas from diagonal of covarience matrix
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x2, fun, prmet, sigma)
#Forcasting emission rate for coming decade
print("Forcasted CO2 Emission")
low, up = err_ranges(2030, fun, prmet, sigma)
print("2030 between", low, "and", up)


#Dataframe containing the data of country India
dt3 = data[(data['Country'] == 'IND')]
print(dt3)
val3 = dt3.values
x3, y3 = val3[:, 1], val3[:, 2]

prmet, cov = opt.curve_fit(fun, x3, y3)
dt3["pop_log"] = fun(x3, *prmet)
print("Parameters are:", prmet)
print("Covariance is:", cov)
plt.plot(x3, dt3["pop_log"], label="Fit")
plt.plot(x3, y3, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.title("CO2 Emission rate in India")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()
plt.savfig("India.png")

#Extracting the sigmas from diagonal of covarience matrix
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x3, fun, prmet, sigma)

#Forcasting emission rate for coming decade
print("Forcasted CO2 Emission")
low, up = err_ranges(2030, fun, prmet, sigma)
print("2030 between", low, "and", up)




