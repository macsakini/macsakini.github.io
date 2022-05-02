Modality and Seasonality on Security Prices in the DOW

The data in the data folder is temporal and thus the models to be used are time series models. The analysis is not geared towards prediction but is focused on finding the right parameters and metrics that best decsribes these datasets. It is also geared by finding which of the time series approaches used in project is more powerful, compact and accurate compared to the other.

![image](https://user-images.githubusercontent.com/47692036/161448320-c6cca985-03b3-4f73-9bdf-7bb12d0ea844.png)
Figure 1.1 The daily returns of the security.


The following is a snippet of the __exponential smoothing__ done in __R__ language:

## Exponential smoothing.
1.	Exponential smoothing. This method works by averaging historical data and using those weights to create a more robust and noise-less trend compared to actual real-time data with high critical volatility. The methodology creates a safe space for predicting future values in a certain trend space. It reduces the effect of the trend on the movement and tries to predict future values.

![image](https://user-images.githubusercontent.com/47692036/161447898-6e00b41d-3827-4d8c-87f7-3526edc83997.png)
Figure 1.2 Fitted ES model.

```r
#Exponential smoothing model
exponential <- function(data){
  summary(es(data,h=100, holdout=FALSE, silent=FALSE, alpha=.2))  
}
exponential(data_x)
```

## ARIMA.
2.	ARIMA. This is a model that is used in finding the flow of data using a regression-like model that is based on previous time stamps. It is a combination of AR or autoregressive and MA moving average models with a integration using differencing to make the non-stationary data stationary. It will be used to show, seasonality, trend and autocorrelation.
![image](https://user-images.githubusercontent.com/47692036/161447913-4d21fa82-7730-48e5-809e-86be642e53f2.png)
Figure 1.3 Arime model fitted on the data.

The following is a snippet of the __ARIMA__ code done in the __R__ language:
```r
#AUTOMATIC ARIMA
autoarima <- function(data){
  x <- auto.arima(data)
  print(checkresiduals(x))
  print(summary(x))
  autoplot(x)
}
autoarima(data_x)

#ARIMA MODEL OF CHOICE
setarima <- function(data){
  x <- Arima(data, order=c(1,1,1))
  print(summary(x))
  print(checkresiduals(x))
  autoplot(x)
}
setarima(data_x)
```

## Regression
3.	Regression. Regression done on a time series perspective has to be dynamic and will show the level of correlation. Regular regression is done using the concerned variable to a relatable variable in same data space. In our scenario since these are parallel time-series, regression can be done directly to the trans-mutated date-time variable or it can be done against the other time-series and the greatest correlation of these can be determined using the coefficients.


**A lot of the data gathered everyday is __time series data__. From individuals, to organizations there has been a fourth industrial revolution of data. Analytics is becoming part of everyday life and slowly by slowly it is being integrated into everyday life. From insurance using telematics to fitness using smart watches, these are just some of the applications that use data and data analysis.**
