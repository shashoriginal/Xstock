import numpy as np
import streamlit as st
import pandas as pd
import requests
from yahoo_fin import stock_info
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Streamlit App Title
st.title('XStock📈: Stock Analysis App')

# Embed TradingView Ticker
st.components.v1.html(
    '''
    <div class="tradingview-widget-container">
     <div class="tradingview-widget-container__widget"></div>
     <div class="tradingview-widget-copyright">
      <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
      </a>
     </div>
     <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
     {
      "symbols": [
       {"proName": "FOREXCOM:SPXUSD", "title": "S&P 500"},
       {"proName": "FOREXCOM:NSXUSD", "title": "US 100"},
       {"proName": "FX_IDC:EURUSD", "title": "EUR to USD"},
       {"proName": "BITSTAMP:BTCUSD", "title": "Bitcoin"},
       {"proName": "BITSTAMP:ETHUSD", "title": "Ethereum"},
       {"description": "INRUSD", "proName": "FX_IDC:INRUSD"}
      ],
      "showSymbolLogo": true,
      "colorTheme": "dark",
      "isTransparent": false,
      "displayMode": "adaptive",
      "locale": "en"
     }
     </script>
    </div>
    ''',
    height=100
)

# User Input for Stock Ticker
ticker = st.text_input('Enter a stock ticker:', 'AAPL').upper()

# Embed TradingView Live Chart
st.components.v1.html(
    f'''
    <div class="tradingview-widget-container">
     <div id="tradingview_{ticker}"></div>
     <div class="tradingview-widget-copyright">
      <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
      </a>
     </div>
     <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
     <script type="text/javascript">
     new TradingView.widget(
     {{
      "width": 600,
      "height": 550,
      "symbol": "{ticker}",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "enable_publishing": false,
      "withdateranges": true,
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "details": true,
      "container_id": "tradingview_{ticker}"
     }}
     );
     </script>
    </div>
    ''',
    height=1300  # Adjusting the height to make the chart dynamic and variable in size
)





try:
    # Fetching historical daily data
    df = stock_info.get_data(ticker)
    
    if df.empty:
        st.error(f"No data available for ticker: {ticker}")
    
    # Calculating SMA (Simple Moving Average)
    df['SMA'] = df['adjclose'].rolling(window=10).mean()
    
    # Calculating EMA (Exponential Moving Average)
    df['EMA'] = df['adjclose'].ewm(span=10, adjust=False).mean()
    
    # Calculating RSI (Relative Strength Index)
    delta = df['adjclose'].diff(1).fillna(0)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=10).mean().fillna(0)
    avg_loss = loss.rolling(window=10).mean().fillna(0)
    rs = avg_gain / avg_loss.replace(0,1)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Displaying the DataFrame with all indicators and time series data
    st.dataframe(df)

except Exception as e:
    st.error(f"Unexpected error fetching stock data: {e}")
    st.error(f"Ticker used: {ticker}")

try:
    # Fetching historical daily data
    df = stock_info.get_data(ticker)

    if df.empty:
        st.error(f"No data available for ticker: {ticker}")

    # 1. Basic Statistical Analysis
    st.write("### Basic Statistical Analysis")
    st.write("#### Descriptive Statistics:")
    st.table(df.describe())

    # 2. Rolling Mean & Standard Deviation
    window_size = st.slider('Select Window Size for Rolling Mean & Std Dev:', min_value=5, max_value=60, value=20, step=1)
    df['Rolling Mean'] = df['adjclose'].rolling(window=window_size).mean()
    df['Rolling Std'] = df['adjclose'].rolling(window=window_size).std()

    st.write("### Stock Price with Rolling Mean and Standard Deviation")
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['adjclose'], label='Adjusted Close')
    plt.plot(df.index, df['Rolling Mean'], label=f'Rolling Mean ({window_size} days)', linestyle='dashed')
    plt.fill_between(df.index, df['Rolling Mean'] - df['Rolling Std'], df['Rolling Mean'] + df['Rolling Std'], color='b', alpha=0.1)
    plt.legend(loc='upper left')
    plt.title('Adjusted Close Price with Rolling Mean & Standard Deviation')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price ($)')
    st.pyplot(plt)

    # 3. Histogram of Daily Price Changes
    st.write("### Histogram of Daily Price Changes")
    daily_changes = df['adjclose'].pct_change().dropna()
    plt.figure(figsize=(10,6))
    sns.histplot(daily_changes, kde=True, bins=50)
    plt.title('Histogram of Daily Price Changes')
    plt.xlabel('Daily Change')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # 4. Correlation Matrix
    st.write("### Correlation Matrix")
    # Selecting only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    # 5. Pair Plot
    st.write("### Pair Plot")
    sns.pairplot(df.dropna())
    st.pyplot(plt)

    # 6. Box Plot
    st.write("### Box Plot of Adjusted Close Prices")
    plt.figure(figsize=(10,6))
    sns.boxplot(df['adjclose'])
    plt.title('Box Plot of Adjusted Close Prices')
    st.pyplot(plt)


    from statsmodels.tsa.seasonal import seasonal_decompose
    st.write("### Time Series Decomposition")
    try:
        decomposed = seasonal_decompose(df['adjclose'].dropna(), period=window_size, model='additive')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,8))
        decomposed.trend.plot(ax=ax1, title='Trend')
        decomposed.seasonal.plot(ax=ax2, title='Seasonal')
        decomposed.resid.plot(ax=ax3, title='Residual')
        plt.tight_layout()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in Time Series Decomposition: {e}")


    # 8. Cumulative Returns
    st.write("### Cumulative Returns Over Time")
    df['Cumulative Return'] = (1 + df['adjclose'].pct_change()).cumprod()
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['Cumulative Return'])
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    st.pyplot(plt)

    # 9. Bollinger Bands
    st.write("### Bollinger Bands")
    df['Bollinger High'] = df['Rolling Mean'] + (df['Rolling Std'] * 2)
    df['Bollinger Low'] = df['Rolling Mean'] - (df['Rolling Std'] * 2)
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['adjclose'], label='Adjusted Close')
    plt.plot(df.index, df['Bollinger High'], label='Bollinger High', linestyle='dashed')
    plt.plot(df.index, df['Bollinger Low'], label='Bollinger Low', linestyle='dashed')
    plt.fill_between(df.index, df['Bollinger Low'], df['Bollinger High'], color='b', alpha=0.1)
    plt.legend(loc='upper left')
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    st.pyplot(plt)

  
    # Date Range Selector
    start_date = st.date_input('Start Date:', value=df.index.min().date(), min_value=df.index.min().date(), max_value=df.index.max().date())
    end_date = st.date_input('End Date:', value=min(start_date, df.index.max().date()), min_value=start_date, max_value=df.index.max().date())
    df = df[start_date:end_date]

    # 11. Volatility Analysis
    st.write("### Volatility Analysis")
    df['Log Return'] = np.log(df['adjclose'] / df['adjclose'].shift(1))
    st.write(f"#### Historical Volatility: {df['Log Return'].std() * np.sqrt(252) * 100:.2f}%")

    # 12. MACD Indicator
    st.write("### MACD Indicator")
    exp1 = df['adjclose'].ewm(span=12, adjust=False).mean()
    exp2 = df['adjclose'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    plt.figure(figsize=(10,6))
    plt.plot(df.index, macd, label='MACD', color='green')
    plt.plot(df.index, signal_line, label='Signal Line', color='red')
    plt.legend(loc='upper left')
    plt.title('MACD Indicator')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    st.pyplot(plt)

    # 13. Stochastic Oscillator
    st.write("### Stochastic Oscillator")
    high_max = df['high'].rolling(window=14).max()
    low_min = df['low'].rolling(window=14).min()
    k = 100 * ((df['adjclose'] - low_min) / (high_max - low_min))
    d = k.rolling(window=3).mean()
    plt.figure(figsize=(10,6))
    plt.plot(df.index, k, label='%K line', color='blue')
    plt.plot(df.index, d, label='%D line', color='orange')
    plt.axhline(80, color='red', linestyle='dashed', alpha=0.7)
    plt.axhline(20, color='green', linestyle='dashed', alpha=0.7)
    plt.fill_between(df.index, y1=20, y2=80, alpha=0.2, color='yellow')
    plt.legend(loc='upper left')
    plt.title('Stochastic Oscillator')
    plt.xlabel('Date')
    plt.ylabel('Value')
    st.pyplot(plt)


    # 14. Interactive Indicator Selection
    st.write("### Interactive Indicator Selection")

    # Create columns for each indicator selection
    cols = st.columns(4)

    # Checkboxes for each indicator
    sma_selected = cols[0].checkbox('SMA', value=True)
    ema_selected = cols[1].checkbox('EMA', value=True)
    bollinger_selected = cols[2].checkbox('Bollinger Bands')
    rsi_selected = cols[3].checkbox('RSI')

    # Initialize the plot
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['adjclose'], label='Adjusted Close')

    # Plot SMA if selected and data is available
    if sma_selected and 'Rolling Mean' in df.columns:
        plt.plot(df.index, df['Rolling Mean'], label='SMA', linestyle='dashed')

    # Plot EMA if selected and data is available
    if ema_selected and 'EMA' in df.columns:
        plt.plot(df.index, df['EMA'], label='EMA', linestyle='dashed')

    # Plot Bollinger Bands if selected and data is available
    if bollinger_selected and 'Bollinger High' in df.columns and 'Bollinger Low' in df.columns:
        plt.plot(df.index, df['Bollinger High'], label='Upper Band', linestyle='dotted')
        plt.plot(df.index, df['Bollinger Low'], label='Lower Band', linestyle='dotted')

    # Plot RSI if selected and data is available, handle potential issues
    if rsi_selected:
        if 'RSI' not in df.columns or df['RSI'].isna().any() or np.isinf(df['RSI']).any():
            st.error("Issue with RSI values!")
        else:
            plt.figure(figsize=(10,6))
            plt.plot(df.index, df['RSI'], label='RSI', linestyle='dotted', color='purple')

    # Set plot labels and show it
    plt.legend(loc='upper left')
    plt.title('Interactive Indicator Selection')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    st.pyplot(plt)




    # Import the required libraries
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.regularizers import l1_l2
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import streamlit as st
    from itertools import product
    import keras_tuner as kt

    def arima_grid_search(data, p_values, d_values, q_values):
        best_score, best_cfg = float("inf"), None
        for p, d, q in product(p_values, d_values, q_values):
            order = (p, d, q)
            try:
                model = ARIMA(data, order=order)
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_score:
                    best_score, best_cfg = aic, order
                    print('ARIMA%s AIC=%.3f' % (order,aic))
            except:
                continue
        return best_cfg

    # Normalize the 'adjclose' for LSTM and keep 'moving_average' unscaled for ARIMA
    scaler = StandardScaler()
    df['scaled_adjclose'] = scaler.fit_transform(df[['adjclose']])

    # Perform ARIMA Grid Search only on 'adjclose' prices
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    best_arima_order = arima_grid_search(df['adjclose'], p_values, d_values, q_values)
    arima_model = ARIMA(df['adjclose'], order=best_arima_order).fit()

    # Split the dataset for train and test
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    train_scaled, test_scaled = df['scaled_adjclose'].iloc[0:train_size], df['scaled_adjclose'].iloc[train_size:len(df)]

    # LSTM Data Preparation
    sequence_length = 60

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data.iloc[i-seq_length:i].values)
            y.append(data.iloc[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)

    # Reshape inputs for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define LSTM Model with Keras Tuner
    def build_lstm_model(hp):
        model = Sequential()
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(LSTM(units=hp.Int('units_' + str(i), min_value=30, max_value=100, step=10),
                        return_sequences=i < hp.Int('num_layers', 1, 3) - 1,
                        input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.05)))
        model.add(Dense(units=1, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                    loss='mean_squared_error')
        return model

    tuner = kt.Hyperband(build_lstm_model,
                        objective='val_loss',
                        max_epochs=10,
                        factor=3,
                        directory='my_dir',
                        project_name='keras_lstm')

    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Making Predictions
    # Predict on the test set
    test_predictions_lstm = model.predict(X_test)
    test_predictions_lstm = scaler.inverse_transform(test_predictions_lstm).flatten()

    # ARIMA predictions for the test set
    arima_predictions = arima_model.predict(start=train_size, end=len(df)-1)
    arima_predictions = arima_predictions.iloc[:len(test_predictions_lstm)].to_numpy()

    # Ensemble Predictions (Simple averaging in this case for test set)
    ensemble_predictions = (arima_predictions + test_predictions_lstm) / 2

    # Plotting the Predictions including the test set
    plt.figure(figsize=(14,7))
    plt.plot(df.index[:train_size], df['adjclose'][:train_size], label='Training Data')
    plt.plot(df.index[train_size:train_size+len(ensemble_predictions)], ensemble_predictions, label='Ensemble Test Predictions', color='orange')
    plt.plot(df.index[train_size:], df['adjclose'][train_size:], label='Actual Prices', color='green')
    plt.title('Stock Price Prediction including Test Set')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Streamlit Display
    st.write("Stock Price Prediction including Test Set:")
    predicted_dates = pd.date_range(start=df.index[train_size], periods=len(test_predictions_lstm), freq='B')
    predictions_df = pd.DataFrame(data=ensemble_predictions, index=predicted_dates, columns=['Ensemble Predicted Prices'])
    st.line_chart(predictions_df)
    st.pyplot(plt)


except Exception as e:
    st.error(f"Error in computational analysis or visualization: {e}")
    



