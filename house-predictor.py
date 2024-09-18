import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the dataset
file_path = 'house_price_prediction_dataset.csv'
data = pd.read_csv(file_path)

# Add 'area_quality' column if not already present
# Example: data['area_quality'] = [7] * len(data) # Replace with actual data

# Prepare the features and target variable
X = data.drop('house_price', axis=1)
y = data['house_price']

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Model accuracy (R² score)
predictions = model.predict(X)
accuracy = r2_score(y, predictions)

# Streamlit App
st.title('Property (House) Price Prediction')

# User inputs
bedrooms = st.number_input('Enter the number of bedrooms', min_value=1, max_value=10, value=2)
bathrooms = st.number_input('Enter the number of bathrooms', min_value=1, max_value=10, value=1)
sqft_living = st.number_input('Enter the square footage', min_value=500, max_value=10000, value=2500)
lot_size = st.number_input('Enter the lot size', min_value=500, max_value=10000, value=2000)
age = st.number_input('Enter the age of the house', min_value=0, max_value=100, value=10)
proximity_to_city_center = st.number_input('Enter the proximity to city center', min_value=1, max_value=100, value=5)
neighborhood_quality = st.number_input('Enter the neighborhood quality', min_value=1, max_value=10, value=5)
area_quality = st.number_input('Enter the area quality', min_value=1, max_value=10, value=7)  # New input

# Currency Conversion Option
currency_option = st.selectbox('Select Currency for Prediction', ['USD', 'INR'])
usd_to_inr_rate = 83.0  # Example conversion rate, adjust as needed

# Prediction
if st.button('Predict House Price'):
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, lot_size, age, proximity_to_city_center,
                                neighborhood_quality, area_quality]],
                              columns=['num_bedrooms', 'num_bathrooms', 'square_footage', 'lot_size', 'age_of_house',
                                       'proximity_to_city_center', 'neighborhood_quality', 'area_quality'])

    prediction = model.predict(input_data)[0]

    # Convert to INR if selected
    if currency_option == 'INR':
        prediction_inr = prediction * usd_to_inr_rate
        st.write(f'Predicted Property Price: ₹{prediction_inr:,.2f}')
    else:
        st.write(f'Predicted Property Price: ${prediction:,.2f}')

# Display model accuracy
st.write(f'Model Accuracy (R² score): {accuracy:.2f}')
