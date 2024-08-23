# **Tools and Libraries that are used in Our Project**

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="https://img.shields.io/badge/Google%20Colab-blue" alt="Google Colab" style="flex: 1 1 30%;">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white" alt="TensorFlow" style="flex: 1 1 30%;">
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white" alt="Keras" style="flex: 1 1 30%;">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white" alt="Pandas" style="flex: 1 1 30%;">
  <img src="https://img.shields.io/badge/Matplotlib-003366?style=flat&logo=matplotlib&logoColor=white" alt="Matplotlib" style="flex: 1 1 30%;">
  <img src="https://img.shields.io/badge/Gradio-FCC624?style=flat&logo=gradio&logoColor=black" alt="Gradio" style="flex: 1 1 30%;">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit" style="flex: 1 1 30%;">
  <img src="https://img.shields.io/badge/yfinance-1f77b4?style=flat&logo=yahoo&logoColor=white" alt="yfinance" style="flex: 1 1 30%;">
  <img src="https://img.shields.io/badge/Scikit%20Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit Learn" style="flex: 1 1 30%;">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy" style="flex: 1 1 30%;">
</div>



# **Stock Market Prediction Using Stacked LSTM**

## **Project Overview**
This project focuses on predicting the stock prices of Alphabet Inc. (Google) using a deep learning model, specifically a **Stacked LSTM** (Long Short-Term Memory) network. By leveraging historical stock data, the model aims to forecast future prices, providing insights that can be valuable for investors and traders.

## **Step-by-Step Process**

### **1. Setting Up the Environment**
The project is developed in **Google Colab**. Necessary Python libraries like **TensorFlow**, **Keras**, **Pandas**, **Matplotlib**, **Gradio**, and **Streamlit** are installed to build, visualize, and deploy the model.

### **2. Data Collection**
Historical stock price data for Alphabet Inc. (Google) is retrieved from **Yahoo Finance**, covering the period from **September 20, 2004, to the present date**.

### **3. Data Preprocessing**
- The data is normalized using **MinMaxScaler** to ensure that all features contribute equally to the model's performance.
- The dataset is split into **training and testing sets** (typically 80-20).
- The data is reshaped into sequences to match the input requirements of the **LSTM model**.

### **4. Model Development**
- A **Stacked LSTM** model is built to handle the time-series nature of stock prices. This involves stacking multiple LSTM layers to capture complex patterns.
- The model is compiled using the **Adam optimizer** and **mean squared error** as the loss function.

### **5. Training the Model**
The model is trained on the training dataset for a specified number of epochs, learning to predict future stock prices based on historical data.

### **6. Model Evaluation**
- The model's predictions are compared with actual stock prices from the testing set to evaluate its accuracy.
- Error metrics like **RMSE** (Root Mean Squared Error) are calculated to quantify the model's performance.

### **7. Future Predictions**
The trained model is used to forecast future stock prices beyond the available data, providing predictions for upcoming months or years.

### **8. Deployment with Gradio and Streamlit**
- **Gradio**: A simple and user-friendly interface is created using Gradio, allowing users to input parameters and get predictions directly from the model.
- **Streamlit**: A more interactive web application is developed using Streamlit, providing a richer user experience with visualizations and dynamic interactions.

### **9. Conclusion**

This project successfully demonstrates the use of Stacked LSTM models for predicting stock prices, offering valuable insights for investors. By integrating Gradio and Streamlit, the model is made accessible and user-friendly, enabling real-time predictions.

### **Important Notes**

- The model effectively captures patterns in historical data, making it a reliable tool for short-term stock price prediction.
- Gradio provides a quick and easy interface, perfect for users who need fast predictions without complex setups.
- Streamlit offers a more comprehensive web application, allowing users to interact with the model through visualizations and dynamic inputs.
- Both deployment methods enhance the usability of the model, making advanced analytics more accessible to a wider audience.

## **Future Work**
- Experimenting with additional features such as **trading volume** or **sentiment analysis**.
- Testing other deep learning models like **GRU** or **CNN**.
- Improving model accuracy through **hyperparameter tuning**.
- Enhancing the **Gradio** and **Streamlit** interfaces with more features and better user experience.

## **How to Run the Project**
1. **Clone the repository**.
2. **Install the required libraries** using the provided `requirements.txt`.
3. **Run the Jupyter Notebook** or the Python script to train the model and make predictions.
4. **Deploy the model** using **Gradio** or **Streamlit** to create a user-friendly interface.

## **Acknowledgments**
- Data sourced from **[Yahoo Finance](https://finance.yahoo.com/quote/GOOG/)**.
- Built with **TensorFlow**, **Keras**, **Gradio**, and **Streamlit**.
