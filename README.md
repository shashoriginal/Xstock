# XStock📈: Stock Analysis App

Welcome to **XStock**, where data meets design to empower your investment journey with real-time insights and in-depth analysis. 🚀

## 🌟 Key Features

- **📊 Live Market Data**: Seamlessly integrated with TradingView to bring you live data for a wide range of instruments.
- **👁️ Interactive Visualization**: Customizable charts with a variety of indicators to dissect the market trends.
- **🧮 Financial Analytics**: SMA, EMA, RSI, and more at your fingertips for thorough stock evaluations.
- **📚 Historical Data Exploration**: Dive into the past performance with detailed historical data and analytics.
- **🔍 Advanced Analysis**: Tools such as Bollinger Bands, MACD, and Stochastic Oscillator to sharpen your strategy.
- **🔮 Future Insights**: Make informed decisions with predictions powered by ARIMA and LSTM models.
- **⚙️ Personalized Experience**: Tailor your analysis with flexible settings and adjustable parameters.
- **💼 Responsive and Intuitive Interface**: A user-friendly platform that's accessible on any device.

## 🚀 Quick Start

Begin your data-driven investment journey with [XStock Live](https://xstock.streamlit.app/)! Please note: the platform may occasionally be under maintenance.

## 📘 Step-by-Step Guide

1. **Ticker Search**: Enter the ticker symbol for a deep dive into the stock's performance.
2. **Watch the Market**: Real-time data and charting provide a snapshot of current market conditions.
3. **Historical Deep Dive**: Assess stock's historical trajectory to spot patterns and trends.
4. **Statistical Overview**: Get a statistical digest of stock's historical volatility and performance.
5. **Data Visualization**: Interpret complex data through visual aids for better understanding.
6. **Trend Analysis**: Unpack the stock data into trend, seasonal, and residual plots for nuanced insights.
7. **Forecasting**: Leverage advanced models to forecast future stock behavior.
8. **Interactive Data Play**: Engage with dynamic charts for a hands-on analytical experience.

## 🖼️ In-Depth Interface Exploration

### Main Dashboard
![Main Dashboard Screenshot](https://i.imgur.com/HuETZlc.png)

The main dashboard serves as your command center, where you can begin your stock analysis by entering the ticker symbol of your choice. It provides a quick overview of market indices and forex rates at a glance.

### Real-Time TradingView Chart

Upon entering a stock ticker, you're presented with a real-time TradingView chart. It displays the stock's price action with options to add various indicators for a multi-dimensional analysis.

### Historical Data Table
![Historical Data Table Screenshot](https://i.imgur.com/IKipAVX.png)

The historical data table is a treasure trove of information, showing the stock's past performance with data points like open, high, low, close, adjusted close, and volume. It also includes computed indicators like SMA and EMA for trend analysis and RSI for measuring overbought or oversold conditions.

### Descriptive Statistics Overview
![Descriptive Statistics Overview Screenshot](https://i.imgur.com/R0OR4i7.png)

This section gives a statistical summary of the stock's data, including count, mean, standard deviation, min, and max values. It provides a quick statistical insight into the stock's volatility and general behavior.

### Stock Price with Rolling Mean and Standard Deviation
![Stock Price with Rolling Mean and Standard Deviation Screenshot](https://i.imgur.com/uhagqKk.png)

Visualize the stock price movement alongside its rolling mean and standard deviation over a selected window period. This helps in identifying the volatility and the moving average trend of the stock price.

### Histogram of Daily Price Changes
![Histogram of Daily Price Changes Screenshot](https://i.imgur.com/ToJyXPN.png)

The histogram offers a visual representation of the frequency distribution of daily price changes. This helps in understanding the variability and the normality of stock price movements.

### Correlation Matrix
![Correlation Matrix Screenshot](https://i.imgur.com/eDp6shd.png)

A correlation matrix is provided to analyze the linear relationship between different financial indicators, such as the rolling mean and standard deviation, and the stock's closing prices.

### Pair Plot
![Pair Plot Screenshot](https://i.imgur.com/LHKSoXJ.png)

The pair plot gives a pairwise relationship visualization across multiple dimensions of the dataset, which is instrumental in spotting correlations and patterns between different variables.

### Box Plot of Adjusted Close Prices
![Box Plot of Adjusted Close Prices Screenshot](https://i.imgur.com/9N7w3Jy.png)

The box plot for adjusted close prices illustrates the distribution of the stock's closing prices, highlighting the median, quartiles, and potential outliers in the data.

### Time Series Decomposition
![Time Series Decomposition Screenshot](https://i.imgur.com/q8WjrkJ.png)

This chart decomposes the stock price into its trend, seasonal, and residual components, providing a clear picture of underlying patterns and irregularities.

### Cumulative Returns Over Time
![Cumulative Returns Over Time Screenshot](https://i.imgur.com/DdaOXmD.png)

The cumulative returns graph showcases the compounded growth of the stock's price over time, offering an insight into the overall investment return potential.


### Bollinger Bands
![Bollinger Bands Screenshot](https://i.imgur.com/eckldDh.png)

The Bollinger Bands chart is an advanced tool for market analysis that shows the stock's adjusted close price along with its volatility. Bollinger Bands consist of:

- **Adjusted Close (Orange Line)**: This line represents the stock's adjusted closing prices over time, reflecting the final price after accounting for any corporate actions.

- **Bollinger High (Red Dashed Line)**: The upper Bollinger Band is plotted two standard deviations above a simple moving average and represents a higher level of volatility or resistance.

- **Bollinger Low (Green Dashed Line)**: The lower Bollinger Band is plotted two standard deviations below the same simple moving average, indicating a lower level of volatility or support.

Together, these bands encapsulate the price movement of a stock, providing a visual representation of its volatility. When the bands are closer together, it indicates a period of lower volatility, and when they are further apart, it suggests higher volatility. Traders often use these bands to predict short-term price movements and to identify overbought or oversold conditions.

### Volatility Analysis, MACD Indicator, Interactive Indicator Selection & Stochastic Oscillator
![Volatility Analysis and MACD Indicator Screenshot](https://i.imgur.com/sERcVcO.png)

This section of the interface provides two critical analytical tools to assess stock behavior:

#### Date Range Selection (Important for Machine Learning)
- **Start Date**: Users can specify the starting point of their analysis, allowing them to focus on a particular period of interest.
- **End Date**: Similarly, the end date can be selected to narrow down the analysis timeframe, providing control over the data being analyzed.

⚠️ **Important Note**: When selecting the date range, ensure that the period is long enough to provide sufficient data for the ARIMA and LSTM forecasting models. These models require a robust dataset to generate accurate predictions and avoid potential errors. Refer to the source code for specific data requirements related to these models.

#### Volatility Analysis
- **Historical Volatility**: This metric, shown as a percentage, indicates the degree of variation or fluctuation in the stock's trading price over the historical period selected. A higher percentage represents higher volatility, which can imply greater risk or potential for a higher return.

### MACD Indicator Detailed View
![MACD Indicator Screenshot](https://i.imgur.com/afkX0Hf.png)

The MACD Indicator chart is a focused view that traders use for identifying changes in the momentum of a stock's price, which can signal potential buy or sell opportunities.

- **MACD Line (Green)**: This is the main line that indicates the momentum of the stock. It's calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA. Rapid movements in the MACD line can suggest trend changes and potential price reversals.

- **Signal Line (Red)**: The signal line is an EMA of the MACD line, typically over 9 periods. Crossovers between the MACD line and the signal line can indicate bullish or bearish entry and exit points. For example, when the MACD line crosses above the signal line, it may be considered a bullish signal, while a cross below could suggest a bearish market sentiment.

The interaction between these two lines is crucial for technical analysts and traders. The MACD line's crossing above the signal line can indicate a time to buy, and crossing below can be a signal to sell. The distance between the MACD and the signal line also reflects the strength of the momentum. A wider gap can suggest stronger momentum and a more robust trend.

### Stochastic Oscillator
![Stochastic Oscillator Screenshot](https://i.imgur.com/OTvaiuw.png)

The Stochastic Oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. The sensitivity of the oscillator to market movements is reducible by adjusting that time period or by taking a moving average of the result.

- **%K line (Blue)**: This line represents the actual value of the stochastic oscillator for each session. It's a measure of the current price relative to the high-low range over a set number of periods. Traditionally, the %K line is drawn as a solid line.

- **%D line (Orange)**: This is a simple moving average of the %K line, which is usually set at three periods. It's drawn as a dashed line and acts as a signal line to the %K line. The %D line helps to identify potential reversals or trend strength.

- **Overbought and Oversold Indicators (Red and Green Horizontal Lines)**: The oscillator ranges from 0 to 100. Typically, readings above 80 are considered overbought, and readings below 20 are considered oversold. These levels can be adjusted based on the security's historical performance.

- **Highlighted Regions**: The chart often highlights the areas where the oscillator indicates overbought or oversold conditions. Overbought conditions can suggest a potential sell opportunity, and oversold conditions can indicate a potential buy opportunity.

Traders use the stochastic oscillator to generate buy and sell signals. A buy signal is suggested when the %K line crosses up through the %D line, especially if the security is in an oversold state. Conversely, a sell signal is suggested when the %K line crosses down through the %D line and the security is in an overbought state.

### Interactive Indicator Selection
![Interactive Indicator Selection Screenshot](https://i.imgur.com/fqen78m.png)

The Interactive Indicator Selection chart offers a dynamic way to visualize multiple technical indicators simultaneously for comprehensive stock analysis.

- **Adjusted Close (Solid Blue Line)**: This line represents the adjusted closing price of the stock, accounting for dividends, splits, and other corporate actions, providing a clear picture of the stock's valuation over time.

- **SMA (Dashed Orange Line)**: The Simple Moving Average (SMA) is plotted to smooth out price data by creating a constantly updated average price over a specific time period, helping to identify trends.

- **Bollinger Bands**:
  - **Upper Band (Dotted Green Line)**: Set two standard deviations above the SMA, this line helps identify potential overbought conditions.
  - **Lower Band (Dotted Red Line)**: Set two standard deviations below the SMA, this line helps identify potential oversold conditions.

- **Indicator Checkboxes**: Users can interactively select or deselect specific indicators like SMA, EMA, Bollinger Bands, and RSI to customize the chart view. This interactivity allows for a tailored analysis experience, focusing on the indicators that are most relevant to the user's strategy.

By using the interactive checkboxes, users can ensure they are selecting the right combination of indicators for their analysis. It's important to note that when choosing the date range for this analysis, it must be sufficient for the ARIMA and LSTM models to function correctly, as insufficient data can lead to errors. Users should reference the code to understand the required data range for these models to ensure a smooth analysis experience.

# 🤖 Machine Learning Predictive Analysis

## 📈 Stock Price Prediction including Test Set

XStock leverages the power of machine learning to offer predictive insights into stock prices, presenting an intuitive visualization of the expected market trends.

- **🔮 Prediction Line (Light Blue)**: This line forecasts the future stock prices using machine learning models. The graph shows the model's output based on its training from historical stock data.

- **📅 Date Range**: The timeline on the x-axis details the span for which the predictions have been made, allowing users to match their investment horizons with the model's insights.

### How It Works
- **ARIMA Model**: Utilizes time series data to understand and predict future points in the series.
- **LSTM Network**: Employs a special kind of neural network that is adept at capturing long-term dependencies in time series data.

![Mind Map: ML](https://i.imgur.com/Qa0Dx1G.png)
![Mind Map: ML1](https://i.imgur.com/G77Gbg1.png)
![Mind Map: ML2](https://i.imgur.com/flvCQ0o.png)
![Mind Map: ML3](https://i.imgur.com/kEYzKvu.png)
![Mind Map: ML4](https://i.imgur.com/Cy6WAVV.png)
![Mind Map: ML5](https://i.imgur.com/Z8Fci98.png)
![Mind Map: ML6](https://i.imgur.com/9NG8q0X.png)

### Best Practices for Users
- **📊 Sufficient Data**: Ensure a robust dataset is provided for the models to make accurate predictions.
- **🛠️ Model Training**: Refer to the application's source code to understand the necessary data range for the ARIMA and LSTM models to function properly.
- **🧪 Test Set Inclusion**: Validate the model's predictions against a test set to gauge accuracy.

The integration of these advanced machine learning models into the XStock platform provides users with a futuristic approach to stock market analysis. By leveraging these models, investors can make more informed decisions based on data-driven forecasts.

# 🎓 Ensemble Learning for Stock Price Prediction

![Ensemble Learning Prediction Screenshot](https://i.imgur.com/cKD4yIu.png)

## 🚀 Stock Price Prediction including Test Set

XStock's ensemble learning model combines multiple machine learning techniques to provide a nuanced forecast of stock prices.

- **📘 Training Data (Dark Blue Line)**: This portion of the graph represents historical data used to train the predictive model. It is crucial for the model to learn the patterns and trends within the stock's price movements.

- **🔮 Ensemble Test Predictions (Orange Line)**: Here, the predictions from different models are combined, or 'ensembled', to generate a more accurate and robust forecast. This line shows the result of that ensemble approach applied to the test set.

- **📉 Actual Prices (Green Line)**: The real stock prices are plotted to provide a point of comparison against the ensemble predictions. This allows users to visually assess the model's performance and accuracy.

### 🧠 Behind the Scenes
- The ensemble method might include different models like ARIMA for understanding the time series components and LSTM for capturing longer-term dependencies.
- The predictions made by individual models are averaged to produce the ensemble prediction, which often leads to improved accuracy over any single model's output.

### ✅ User Guidelines
- **🔍 Adequate Data Input**: For the ensemble model to work effectively, ensure that a comprehensive dataset is available for both training and testing phases.
- **🛠️ Accurate Training**: Consult the source code to verify the data requirements for the ARIMA and LSTM models, as they form the backbone of the ensemble approach.
- **📊 Error Margin**: While the ensemble predictions are typically more reliable, always consider the potential margin of error when making investment decisions based on these forecasts.

The Ensemble Learning feature of XStock represents a sophisticated approach to stock market forecasting, offering users a composite analytical tool that draws from the strengths of multiple predictive algorithms.

👤 **Author**: [Shashank Raj](https://github.com/shashoriginal)
