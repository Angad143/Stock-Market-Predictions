import streamlit as st 
import yfinance as yf  
import numpy as np  
import pandas as pd  
from tensorflow.keras.models import load_model  
from sklearn.preprocessing import MinMaxScaler  
import matplotlib.pyplot as plt 

# Import io to handle image data.
import io  

# Load the saved model from a specified path.
model = load_model("lstm_model.h5")

# Define a function to predict stock prices.
def predict_stock_prices(ticker, days):
    # Define the start and end dates for fetching historical stock data.
    end = pd.Timestamp.today()  
    start = pd.Timestamp(end.year - 20, end.month, end.day) 

    # Fetch historical stock data for the given ticker.
    data = yf.download(ticker, start, end=end)
    adj_close_prices = data['Adj Close'].values.reshape(-1, 1) 

    # Normalize the stock price data.
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_prices = scaler.fit_transform(adj_close_prices)  

    # Create sequences for LSTM model input.
    sequence_length = 60  # Define the length of input sequences for LSTM.
    last_sequence = scaled_prices[-sequence_length:]  

    # Predict future stock prices.
    # Initialize a list to store future predictions.
    future_predictions = []  
    for _ in range(days):
        next_price_scaled = model.predict(last_sequence.reshape(1, sequence_length, 1))[0, 0]  
        future_predictions.append(next_price_scaled) 
        last_sequence = np.append(last_sequence[1:], next_price_scaled) 

    # Reverse the scaling of the predicted prices.
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))  

    # Create a date range for the future predictions.
    future_dates = pd.date_range(start=data.index[-1], periods=days + 1, inclusive='right')  

    # Prepare the result DataFrame with predicted prices and dates.
    result = pd.DataFrame({
        'Date': future_dates, 
        'Predicted Prices': future_predictions.flatten() 
    })

    # Plot the results.
    plt.figure(figsize=(14, 7)) 
    plt.plot(data.index, adj_close_prices, color='blue', label='Actual Stock Price')  
    plt.plot(result['Date'], result['Predicted Prices'], color='purple', label='Future Predictions') 
    plt.xlabel('Date')  
    plt.ylabel('Stock Price')  
    plt.legend() 
    plt.title(f'{ticker} Stock Price Prediction') 
    plt.grid(True)  
    
    # Save the plot to a BytesIO object.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return result, buf  # Return the DataFrame and the BytesIO object containing the plot image.

# Streamlit app
st.title("Stock Market Prediction")  # Title of the Streamlit app.
st.write("Predict future stock prices using an LSTM model. Enter a stock ticker and select the number of days to predict.")

# Inputs
ticker = st.text_input("Stock Ticker")  # Input for stock ticker symbol.
days = st.slider("Days to Predict", min_value=1, max_value=50, value=20)  # Slider for selecting number of days to predict.

# Button to trigger prediction
if st.button("Predict"):
    if ticker:
        # Call the prediction function
        result, plot_img = predict_stock_prices(ticker, days)

        # Display results
        st.write("Predicted Stock Prices:")
        st.dataframe(result)  # Display the DataFrame with predicted stock prices.

        # Display the plot
        st.image(plot_img, caption=f"{ticker} Stock Price Prediction")  # Display the plot image.
    else:
        st.error("Please enter a valid stock ticker.")  # Display an error message if no ticker is entered.
