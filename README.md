# Vapour Pressure Forecasting App 🌤️

This is a Streamlit web application for time series forecasting of vapour pressure using machine learning and statistical models.

## 📂 Project Structure

- `app.py` – Main Streamlit application
- `requirements.txt` – List of dependencies
- `cleaned_weather.csv` – (Optional) Sample dataset for testing

## 📊 Features

- Upload your own CSV dataset
- Visualize raw time series
- Perform time series decomposition (Additive/Multiplicative)
- Generate forecasts using:
  - ARIMA
  - ETS (Exponential Smoothing)
  - Prophet
- View evaluation metrics (MAE, RMSE, MAPE)

## 🚀 Deployment

This app is deployed on [Streamlit Cloud](https://streamlit.io/cloud).  
To deploy your own version:

1. Fork this repository or clone it:
    ```bash
    git clone https://github.com/your-username/vapour-pressure-forecast-app.git
    cd vapour-pressure-forecast-app
    ```

2. Add your own dataset (optional).

3. Push to GitHub and deploy on Streamlit Cloud.

## ▶️ Run Locally

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Requirements

- Python 3.8+
- Packages listed in `requirements.txt`

## 📬 Contact

Feel free to reach out if you have questions or want to contribute!
