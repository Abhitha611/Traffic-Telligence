from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature names
model = joblib.load('traffic_model.pkl')
features = joblib.load('model_features.pkl')

@app.route('/')
def home():
    return render_template('Smart.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form data
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hours'])
        minute = int(request.form['minutes'])
        second = int(request.form['seconds'])
        holiday = request.form['holiday']  # Optional - not used here

        dayofweek = pd.Timestamp(f"{year}-{month}-{day}").dayofweek

        input_data = {
            'temp': temp,
            'rain_1h': rain,
            'snow_1h': snow,
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'dayofweek': dayofweek
        }

        # Add weather dummy columns
        for col in features:
            if col.startswith("weather_main_"):
                input_data[col] = 1 if col == f'weather_main_{weather}' else 0

        # Ensure all features present
        df_input = pd.DataFrame([input_data])
        df_input = df_input.reindex(columns=features, fill_value=0)

        prediction = model.predict(df_input)[0]
        return render_template('Smart.html', prediction_text=f"Predicted Traffic Volume: {int(prediction)} vehicles")

    except Exception as e:
        return render_template('Smart.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
