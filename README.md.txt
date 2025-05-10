

# Machine Condition Prediction using Random Forest

**SUPRAKASH MAITY**
**2nd Year, Mechanical Engineering**
**ARM College of Engineering & Technology**
**Course: Data Analysis in Mechanical Engineering**

---

## Project Overview

This project focuses on predicting the operational condition of industrial machines using a trained **Random Forest Classifier**. By analyzing key parameters such as temperature, vibration, RPM, and oil quality, the model helps determine whether a machine is functioning normally or may require maintenance.

This approach can be very useful in real-world maintenance systems where early detection of machine faults can save time and resources.

---

## What You Need to Get Started

Before running the code, make sure you install the required Python libraries by executing the command below in your terminal:

```
pip install -r requirements.txt
```

This will set up all the necessary packages for model loading, data handling, and predictions.

---

## Files Included for Prediction

Ensure the following files are present in your project folder:

* `random_forest_model.pkl`: The trained machine learning model.
* `scaler.pkl`: Scaler used to standardize the input data.
* `selected_features.pkl`: Contains the list of features used while training the model.

These files help maintain consistency between training and prediction stages.

---

## Step-by-Step Prediction Process

1. **Load Required Files**

   * Load the trained model, scaler, and feature list using `joblib`.

2. **Prepare Your Input**

   * Create a DataFrame with one row, including all the necessary machine features.
   * Make sure all feature names exactly match those used during training.

3. **Normalize the Input**

   * Use the scaler to transform the input data so it fits the distribution used during training.

4. **Make a Prediction**

   * Use the model’s `predict()` method to classify the condition.
   * Use `predict_proba()` to view the confidence levels for each class.

---

## Sample Code to Run Predictions

Here is a simple script to run a prediction using your input values:

```python
import joblib
import pandas as pd

# Load trained components
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input data
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Align feature order
new_data = new_data[selected_features]

# Normalize data
scaled_data = scaler.transform(new_data)

# Generate predictions
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Condition:", prediction[0])
print("Probability Scores:", prediction_proba[0])
```

---

## Important Reminders

* The input must include **all features** used during training, and the **feature order** must be correct.
* Input values should fall within the range observed during the model's training phase.
* Modifying the feature names or order can lead to incorrect predictions.

---

## Updating the Model (Optional)

If you want to retrain the model with new data:

* Use the same preprocessing steps for consistency.
* After training, update and save the new model, scaler, and feature list using `joblib`.
* Make sure the new data is properly scaled and structured like the original training data.

---

## Real-World Applications

* Predicting machine health in industrial settings.
* Early detection of faulty operations in manufacturing lines.
* Integrating with IoT sensors for live monitoring and predictive maintenance.

