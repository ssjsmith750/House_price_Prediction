import streamlit as st
import pandas as pd
import joblib

# Load the Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Load the dataset
df = pd.read_excel(r"C:\Users\ssjsm\OneDrive\Desktop\GUVI\house_pridiction_Task\House_Rent_Train.xlsx")

# Create Streamlit app
st.title("Property Rent Prediction App")

# Create dropdowns for each feature
property_type = st.selectbox("Select Property Type", df['type'].unique())
locality = st.selectbox("Select Locality", df['locality'].unique())
lease_type = st.selectbox("Select Lease Type", df['lease_type'].unique())
furnishing = st.selectbox("Select Furnishing", df['furnishing'].unique())
parking = st.selectbox("Select Parking", df['parking'].unique())
facing = st.selectbox("Select Facing", df['facing'].unique())
water_supply = st.selectbox("Select Water Supply", df['water_supply'].unique())
building_type = st.selectbox("Select Building Type", df['building_type'].unique())

# Create input fields for numerical features
latitude = st.number_input("Enter Latitude", value=df['latitude'].mean())
longitude = st.number_input("Enter Longitude", value=df['longitude'].mean())
gym = st.checkbox("Gym")
lift = st.checkbox("Lift")
swimming_pool = st.checkbox("Swimming Pool")
negotiable = st.checkbox("Negotiable")
property_size = st.number_input("Enter Property Size", value=df['property_size'].mean())
property_age = st.number_input("Enter Property Age", value=df['property_age'].mean())
bathroom = st.number_input("Enter Number of Bathrooms", value=df['bathroom'].mean())
cup_board = st.number_input("Enter Cupboards", value=df['cup_board'].mean())
floor = st.number_input("Enter Floor", value=df['floor'].mean())
total_floor = st.number_input("Enter Total Floors", value=df['total_floor'].mean())
balconies = st.number_input("Enter Number of Balconies", value=df['balconies'].mean())

# Create checkboxes for amenities
amenities = [st.checkbox(amenity) for amenity in df['amenities'].unique()]

# Combine user inputs into a DataFrame for prediction
user_input = pd.DataFrame({
    'type': [property_type],
    'locality': [locality],
    'lease_type': [lease_type],
    'latitude': [latitude],
    'longitude': [longitude],
    'gym': [1 if gym else 0],
    'lift': [1 if lift else 0],
    'swimming_pool': [1 if swimming_pool else 0],
    'negotiable': [1 if negotiable else 0],
    'furnishing': [furnishing],
    'parking': [parking],
    'property_size': [property_size],
    'property_age': [property_age],
    'bathroom': [bathroom],
    'facing': [facing],
    'cup_board': [cup_board],
    'floor': [floor],
    'total_floor': [total_floor],
    'water_supply': [water_supply],
    'building_type': [building_type],
    'balconies': [balconies],
    'rent': [0]  # Placeholder for the target variable, as it's not needed for prediction
})

# Make prediction using the Random Forest model
prediction = rf_model.predict(user_input.drop('rent', axis=1))

# Display the predicted rent
st.subheader(f"Predicted Rent: {prediction[0]:,.2f} INR")
