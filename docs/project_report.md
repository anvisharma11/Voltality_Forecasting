# S&P 500 Volatility Forecasting: Econometric GARCH Family vs. Deep Recurrent Neural Networks

Anvi Sharma-240958190,
Tanvika Shetty 240958001,
CSFT A,
AF IA-3
**Codebase:** [GitHub Repository Link](https://github.com/anvisharma11/Voltality_Forecasting)  
**Data Source:** [Dataset Source Link (Yahoo Finance ^GSPC)](https://finance.yahoo.com/quote/%5EGSPC)  
**Execution Environment:** Python 3.12, PyTorch CPU (`torch`), `arch`, `statsmodels`, `scipy`

---

## 1. Introduction

Volatility tells us how much the price of something goes up and down. It matters a lot in finance. Banks use it to decide how much money to keep aside, and it is also used in risk management, in pricing derivatives and in setting margin. Measures like Value-at-Risk (VaR) and Expected Shortfall (ES) need a volatility forecast to work. If we forecast it too low, we can lose more money than we expected. If we forecast it too high, we keep extra cash aside that we did not need.

The most common way to forecast volatility is to use GARCH models. There are three main types, GARCH, EGARCH and GJR-GARCH, and people have been using them for about 40 years. They work well because they capture two things we see in real market data. The first is volatility clustering. This means that after a big price move, another big move usually follows, and after a calm day, the next days are usually calm too. The second is the leverage effect. This means that when prices fall, volatility goes up more than when prices rise by the same amount.

But these models also have some problems. They follow a fixed formula, and they assume the errors follow a normal or Student’s t distribution. They also only use the close-to-close return of each day. So they ignore the high and low price of the day and the trading volume, even though these can tell us a lot about what the market is doing.

```
       Closed-form Return Dynamics                  Non-Linear Multi-Feature Dynamics
      ┌───────────────────────────┐                 ┌───────────────────────────────┐
      │  Close-to-Close Returns   │                 │ Returns + High-Low Spreads    │
      │         r(t) only         │                 │     + Volume Surges           │
      └─────────────┬─────────────┘                 └───────────────┬───────────────┘
                    │                                               │
                    ▼                                               ▼
       ┌─────────────────────────┐                   ┌─────────────────────────────┐
       │   Parametric GARCH/     │                   │  Deep Sequence Model (LSTM) │
       │     EGARCH / GJR        │                   │     30-Day Receptive Field  │
       └────────────┬────────────┘                   └──────────────┬──────────────┘
                    │                                               │
                    └───────────────► 5-Day Realized ◄──────────────┘
                                     Volatility Forecast
```

So we also tried an LSTM (Long Short-Term Memory) network, which is a type of neural network that works well with data over time. We do not have to give it a formula. It learns the pattern from past data on its own. We can also give it many inputs together, like returns, price range and volume. But it has its own problems. It can give different results each time we train it, it has a lot of parameters, and it can overfit, which means it learns the training data too well and then does badly on new data.

### Project Scope and Core Objectives

In our project we compared the GARCH models with an LSTM using the same data, S&P 500 index (^GSPC) from January 2000 till December 2021, around 22 years. We are forecasting 5-day annualized realized volatility (rv5_t). The test period is the last 505 days, so 2020 and 2021, and we chose this because it has the COVID-19 crash in it and also the recovery after that, so it is a proper test for the models.

What we wanted to do was this:

1. First we wanted to check if volatility clustering is really there in the data. For that we ran Engle’s ARCH-LM test and the Jarque-Bera test, and after that we fitted GARCH(1,1), EGARCH(1,1) and GJR-GARCH(1,1) using maximum likelihood with Student’s t errors, because returns have fat tails.
2.For the LSTM we made inputs using the past 30 days, with three features which are the daily log return, the intraday range and the percentage change in volume. We were careful about lookahead bias here, so the scalers were fitted on the training data only and then used on the validation and test data, otherwise the model would be using information from the future.
3. We built the LSTM in PyTorch with early stopping. Since one run can give a different answer from another, we trained it with three seeds (42, 43 and 44) and took the ensemble of them.
4. In the end we compared everything against a simple persistence benchmark, which is just rv5 of the previous day, using RMSE, MAE and QLIKE loss. We also looked at the results separately for the stress period (March 2020) and the calm period (2021), and used the Diebold-Mariano test to see if the difference between the models is actually significant or not.

---

## 2. Related Work

Many papers are already there on volatility forecasting. First people used econometrics models like ARCH and GARCH. Later they started using machine learning also. We read 15 papers and made Table 1. For each paper we wrote what they did and what they did not do. What they did not do is the gap. We try to cover these gaps in our project.

### Comprehensive Literature Review Table

| # | Authors & Year | Paper Title | Core Contribution | Research Gap Addressed in Our Work |
|---|---|---|---|---|
| 1 | Engle (1982) | *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation* | First ARCH model. It showed variance is not constant and changes with time. High volatility days come together | Good and bad returns are treated same. Fat tails not handled. Cant use volume or other features |
| 2 | Bollerslev (1986) | *Generalized Autoregressive Conditional Heteroskedasticity* | Added past variance in ARCH and made GARCH($p,q$). It needs less parameters | Good news and bad news have same effect. Need $\alpha_1 + \beta_1 < 1$ for stationarity |
| 3 | Nelson (1991) | *Conditional Heteroskedasticity in Asset Returns: A New Approach* | Made EGARCH using $\ln(\sigma_t^2)$. It can show leverage effect and there is no sign restriction on parameters | Estimation becomes unstable in sudden big shocks like the COVID-19 crash |
| 4 | Glosten, Jagannathan, & Runkle (1993) | *On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks* | Made GJR-GARCH. One extra term is added so negative shocks get more weight | Still a fixed linear model. Cant use high-low range or volume properly |
| 5 | Andersen & Bollerslev (1998) | *Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts* | Showed squared daily returns give wrong errors when we judge models. Told to use realized volatility from high frequency data | Used intraday squared returns only. Not done for neural networks on multi-day realized volatility |
| 6 | Hansen & Lunde (2005) | *A Forecast Comparison of Volatility Models: Does Anything Beat a GARCH(1,1)?* | Compared 330 models on exchange rates and IBM returns. GARCH(1,1) was hard to beat on exchange rates | Only parametric models and normal market period. No deep learning and no pandemic |
| 7 | Patton (2011) | *Volatility Forecast Comparison Using Imperfect Volatility Proxies* | Proved QLIKE and MSE loss are safe to use even if the volatility proxy has noise | Mostly theory. Not tested for neural networks vs asymmetric GARCH |
| 8 | Diebold & Mariano (1995) | *Comparing Predictive Accuracy* | Gave Diebold-Mariano (DM) test to check if one forecast is really better than other | Many machine learning papers skip this test. They show only lower RMSE and dont check significance |
| 9 | Hochreiter & Schmidhuber (1997) | *Long Short-Term Memory* | Gave LSTM cell with input, forget and output gates. It solves vanishing gradient problem for long sequences | General machine learning paper. Not told which window length or features are good for finance |
| 10 | Kim & Won (2018) | *Forecasting the Volatility of Stock Price Index: A Hybrid Model Integrating LSTM with Multiple GARCH-Type Models* | Used GARCH outputs as input to LSTM for KOSPI 200 volatility | Two training steps so first step error goes to second step. Simple LSTM on raw price features not tried |
| 11 | Bucci (2020) | *Realized Volatility Forecasting with Neural Networks* | Tried feedforward and recurrent networks on S&P 500 realized volatility. Found they handle long memory better than HAR-RV | Volume not used. Crisis period and after crisis period not compared |
| 12 | Liu (2019) | *Novel Volatility Forecasting Using Deep Learning-Long Short Term Memory Recurrent Neural Networks* | Used LSTM on equity index volatility. Said it is better than normal linear autoregressive models | No persistence baseline ($rv_{t-1}$). Model trained only once so random weights effect not checked |
| 13 | Christensen, Dahl, & Iglesias (2012) | *Semiparametric Inference in a GARCH-in-Mean Model* | Showed that wrong error distribution gives very biased parameter estimates | Still parametric. Neural networks which dont need this assumption not tried |
| 14 | Arnerić, Poklepović, & Aljinović (2014) | *GARCH Based Artificial Neural Networks in Forecasting Volatility* | Joined GARCH variance with small neural networks to model kurtosis better | Only shallow networks with one layer. No modern sequence models. No multiple seeds |
| 15 | Lim & Zohren (2021) | *Time-Series Forecasting with Deep Learning: A Survey* | Survey of deep learning models for time series. Explains why recurrent gating and inductive bias are important | Very broad survey. No direct comparison with re-estimated asymmetric GARCH on equity indices |

After reading these papers we saw some things come again and again. In GARCH papers (1 to 4 and 13) the model keeps improving but all use only returns and a fixed formula. In neural network papers (10 to 12 and 14) deep learning gives good results but many train only once. Some dont use persistence baseline. Some dont test on a crisis period. The papers on evaluation (5, 7 and 8) tell how to compare forecasts properly but not many people use them with neural networks. So we try to fill these gaps in our project. We use a simple LSTM with returns, range and volume as input. We train it with three seeds. Then we compare it with three GARCH models and a persistence benchmark. We use QLIKE and Diebold-Mariano test on the COVID period.

---

## 3. Methodology
This project's methodology compares deep learning methods with conventional statistical methods for stock market volatility prediction.We used historical S&P 500 index data for this purpose. The GARCH model and the LSTM model are the two primary forecasting techniques used in the project.The GARCH model estimates future variance based on historical market returns and volatility trends. In contrast, the LSTM model looks for patterns in past market data sequences that could be useful in forecasting future volatility.The entire procedure included gathering the data, processing and cleaning it, creating the forecasting models, producing forecasts, and assessing how well they worked.Understanding how these two methods function with actual financial market data, especially during times of shifting market conditions, was the primary goal.
DataSet: The dataset used in this project consists of historical S&P 500 index data collected from Yahoo Finance using the yfinance Python library. The S&P 500 Index was represented by the ticker symbol ^GSPC.The information spans January 2000 to December 2021.
The following data is included in the dataset:

• Open: The price at which the index began day-to-day trading.

• High: The maximum amount attained throughout the trading day.

• Low: The lowest amount attained throughout the trading day.

• Close: The index's closing value.

To prepare the data for model building, we preprocessed it after it was collected.We computed daily logarithmic returns rather than utilizing stock prices directly. Returns are frequently employed in financial forecasting and offer a better knowledge of the percentage fluctuations in the market. 
The daily log return was calculated using:
 

We also created additional features to capture different aspects of market behaviour.
1. High-Low Price Range
This feature measures the difference between the highest and lowest prices during a trading day.
 
2. Volume Percentage Change
This feature measures changes in trading volume compared to the previous trading day.
 
3. Five-Day Realized Volatility
We also calculated five-day realized volatility using the squared daily returns over a rolling five-day period.

 
This feature provides an indication of recent market volatility.

Prior to creating the forecasting models, we used the following tools to analyze the statistical properties of the financial data: 
The ARCH-LM Test:
The presence of volatility clustering in the returns was determined using the ARCH-LM test.Changes in market volatility were not entirely random, according to the test results, which revealed an extremely substantial ARCH effect.To put it simply, times of high volatility were frequently followed by other periods of high volatility, whilst calmer periods were typically followed by rather calm market conditions. 
The training, validation, and testing portions of the dataset were separated chronologically.Financial data follows a time sequence, therefore random splitting was avoided. Inaccurate evaluation could result from training a model that forecasts the past using future observations.The dataset was divided as follows:
Dataset	Period	Observations
Training	2000–2017	4,523
Validation	2018–2019	503
Testing	2020–2021	505

Implementation of the GARCH Model
The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model was implemented to forecast future market variance using historical returns.
We used a GARCH (1,1) model with a constant mean and Student's t-distribution.
The GARCH (1,1) model is represented as:
 
The Python arch library was used to implement the model.The training dataset, which included 4,523 observations, was first used to train the model. Forecasts for the testing and validation phases were then produced one step ahead of time.The rolling forecasting approach was a key component of the implementation. Before including the actual return for that day, the model produced a forecast. The actual return was added to the historical data once it became available.Using the available historical observations, the model parameters were adjusted every 20 forecasts. This made it easier for the model to adapt to shifting market conditions.Additionally, as financial returns may exhibit dramatic fluctuations, we employed Student's t-distribution rather than assuming normally distributed errors.The forecasts that were produced were saved in CSV format for additional analysis.
Implementation of the LSTM Model
In addition to the GARCH model, we integrated an LSTM neural network for market variance forecasting. 
LSTM is a kind of recurrent neural network built to learn patterns from sequential data. The relationship between current and previous observations makes this especially helpful. 
For this initiative, the LSTM model employed four input variables: 
1. Everyday log return values. 
2. The span between the highest and lowest prices. 
Percentage change in trading volume. 
A five-day realized volatility measure. 
Prior to training, the input features were made zero mean and unit variance via StandardScaler. To prevent using data from future observations during preprocessing, the scaler was fitted solely with the training data.
The model used the previous 30 trading days as its input sequence to predict the current day's variance, represented by the squared daily return.
The LSTM architecture consisted of:
•	Input Layer: Accepts sequences containing 30 trading days and four features.
•	LSTM Layer: Contains 64 units to learn patterns from the historical sequences.
•	Dropout Layer: Uses a dropout rate of 0.20 to reduce overfitting.
•	Dense Layer: Produces the final predicted value.
The model was trained using the Adam optimizer with Mean Squared Error (MSE) as its loss function.
The training settings included a batch size of 32 and a maximum of 100 epochs. Early stopping with a patience of 10 epochs was used to stop training when validation performance stopped improving.





## 4. Results and Discussion
We evaluated GARCH and LSTM model performance with the historical S&P 500 dataset after development and training. 
The primary goal of this phase was to assess the models' accuracy in forecasting market variance and the differences in their performance. 
The same validation and testing periods were used to evaluate both models. This allowed us to assess their forecasting accuracy with a standardized method. 
The testing phase spanned the years 2020 and 2021, covering the COVID-19 market crash and the recovery that followed. 
This proved especially valuable, as it enabled us to evaluate model performance during both significant market uncertainty and calmer conditions. 
GARCH Model Results
The GARCH model was evaluated separately on the validation and testing datasets. The model employed past returns to calculate conditional variance and produced forecasts via a rolling method. The parameters were updated after every 20 forecasts.  The results obtained were:
Evaluation Metric	Validation	Testing
QLIKE	0.8780	1.7222
MSE	4.3483	155.4428
RMSE	2.0853	12.4677
MAE	1.0394	3.8518
The validation results revealed a much better GARCH model performance than the testing phase outcomes. 
Testing revealed more errors, showing the model struggled to predict volatility during unpredictable market conditions. 
This may be linked to the abrupt market shifts seen during the 2020 pandemic.

LSTM Model Results
The LSTM model underwent training with random seeds 42, 43 and 44. 
Models trained separately had their predictions combined by averaging to form the final LSTM ensemble. 
The training also used early stopping to prevent unnecessary training and lower overfitting risk. The results obtained were:
Evaluation Metric	Validation	Testing
QLIKE	4.9980	9.4196
MSE	4.4721	116.8520
RMSE	2.1147	10.8098
MAE	0.8141	2.4380
The results show that the LSTM model maintained relatively consistent error levels across the validation and testing periods for RMSE and MAE, although its QLIKE score increased during testing.
The LSTM model performed better in terms of RMSE and MAE during the testing period compared to the GARCH model.
Challenges and Observations
We had a few observations during the implementation and evaluation:
1. Preprocessing financial data
Due to the nature of financial data it can contain extreme movements, missing data and changing volatility. So preprocessing was required.
2. No future data leakage
As it is a time series problem we needed to make sure we weren’t predicting past values using future data.
We split the data into chronological parts. Then the GARCH model would produce the prediction and we added the real return for that day.
3. Overfitting
To reduce overfitting in the LSTM model we implemented early stopping and dropout regularisation. We also trained 3 models using different random seeds.
4. Different Evaluation Metrics
We saw that the model achieving the lowest RMSE didn’t always achieve the lowest QLIKE. This helped us realise how important it is to look at various performance metrics when looking at volatility predictions.


## 5. References

1. Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International Economic Review*, 39(4), 885–905. https://doi.org/10.2307/2527343
2. Arnerić, J., Poklepović, T., & Aljinović, Z. (2014). GARCH based artificial neural networks in forecasting volatility. *Croatian Operational Research Review*, 5(2), 245–259. https://doi.org/10.17535/crorr.2014.0011
3. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327. https://doi.org/10.1016/0304-4076(86)90063-1
4. Bucci, A. (2020). Realized volatility forecasting with neural networks. *Journal of Financial Econometrics*, 18(4), 688–731. https://doi.org/10.1093/jjfinec/nbaa008
5. Christensen, B. J., Dahl, C. M., & Iglesias, E. M. (2012). Semiparametric inference in a GARCH-in-mean model. *Journal of Econometrics*, 167(2), 409–420. https://doi.org/10.1016/j.jeconom.2011.09.020
6. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599
7. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007. https://doi.org/10.2307/1912773
8. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *The Journal of Finance*, 48(5), 1779–1801. https://doi.org/10.1111/j.1540-6261.1993.tb05128.x
9. Hansen, P. R., & Lunde, A. (2005). A forecast comparison of volatility models: does anything beat a GARCH(1,1)?. *Journal of Applied Econometrics*, 20(7), 873–889. https://doi.org/10.1002/jae.800
10. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
11. Kim, H. Y., & Won, C. H. (2018). Forecasting the volatility of stock price index: A hybrid model integrating LSTM with multiple GARCH-type models. *Expert Systems with Applications*, 103, 25–37. https://doi.org/10.1016/j.eswa.2018.02.040
12. Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. *Philosophical Transactions of the Royal Society A*, 379(2194), 20200209. https://doi.org/10.1098/rsta.2020.0209
13. Liu, Y. (2019). Novel volatility forecasting using deep learning-long short term memory recurrent neural networks. *Expert Systems with Applications*, 132, 99–109. https://doi.org/10.1016/j.eswa.2019.04.038
14. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347–370. https://doi.org/10.2307/2938260
15. Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256. https://doi.org/10.1016/j.jeconom.2010.03.034

---

## 6. Code and Dataset Access

- **GitHub Source Code Repository:**  
  [https://github.com/anvisharma11/Voltality_Forecasting](https://github.com/anvisharma11/Voltality_Forecasting)  
  *Contains the complete implementation including `data.py`, `garch_models.py`, `lstm_data.py`, `lstm_model.py`, `evaluate.py`, `metrics.py`, unit test suites, and the interactive Streamlit dashboard (`app.py`).*

- **Raw Financial Market Dataset:**  
  [Yahoo Finance S&P 500 Historical Ticker (^GSPC)](https://finance.yahoo.com/quote/%5EGSPC)  
  *Continuous daily Open, High, Low, Close, and Volume records from 2000-01-03 to 2021-12-31.*

