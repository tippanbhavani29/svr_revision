# =========================================
# IMPORT LIBRARIES
# =========================================

import streamlit as st
import numpy as np
import pickle

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="House Price Prediction - SVR",
    page_icon="🏠",
    layout="centered"
)

# =========================================
# LOAD MODEL & SCALERS
# =========================================

model = pickle.load(
    open("svr_model.pkl", "rb")
)

scaler_X = pickle.load(
    open("scaler_X.pkl", "rb")
)

scaler_y = pickle.load(
    open("scaler_y.pkl", "rb")
)

# =========================================
# TITLE
# =========================================

st.title("🏠 House Price Prediction using SVR")

st.write(
    """
    This application predicts house prices
    using Support Vector Regression (SVR).
    """
)

# =========================================
# USER INPUTS
# =========================================

st.header("Enter House Details")

# Taxi Distance

Taxi_dist = st.number_input(
    "Taxi Distance",
    min_value=0.0,
    value=5.0
)

# Market Distance

Market_dist = st.number_input(
    "Market Distance",
    min_value=0.0,
    value=3.0
)

# Hospital Distance

Hospital_dist = st.number_input(
    "Hospital Distance",
    min_value=0.0,
    value=2.0
)

# Carpet Area

Carpet_area = st.number_input(
    "Carpet Area",
    min_value=100.0,
    value=1000.0
)

# Builtup Area

Builtup_area = st.number_input(
    "Builtup Area",
    min_value=100.0,
    value=1200.0
)

# Rainfall

Rainfall = st.number_input(
    "Rainfall",
    min_value=0,
    value=100
)

# =========================================
# CATEGORICAL INPUTS
# =========================================

# Parking Type

Parking_type = st.selectbox(
    "Parking Type",
    [
        "Open",
        "Covered",
        "No Parking",
        "Not Provided"
    ]
)

# City Type

City_type = st.selectbox(
    "City Type",
    [
        "CAT A",
        "CAT B",
        "CAT C"
    ]
)

# =========================================
# MANUAL ONE-HOT ENCODING
# =========================================

# Parking Type Encoding

Parking_type_No_Parking = 0
Parking_type_Not_Provided = 0
Parking_type_Open = 0

if Parking_type == "No Parking":

    Parking_type_No_Parking = 1

elif Parking_type == "Not Provided":

    Parking_type_Not_Provided = 1

elif Parking_type == "Open":

    Parking_type_Open = 1

# =========================================
# City Type Encoding
# =========================================

City_type_CAT_B = 0
City_type_CAT_C = 0

if City_type == "CAT B":

    City_type_CAT_B = 1

elif City_type == "CAT C":

    City_type_CAT_C = 1

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button("Predict House Price"):

    # =====================================
    # CREATE FEATURE ARRAY
    # =====================================

    features = np.array([[
        Taxi_dist,
        Market_dist,
        Hospital_dist,
        Carpet_area,
        Builtup_area,
        Rainfall,
        Parking_type_No_Parking,
        Parking_type_Not_Provided,
        Parking_type_Open,
        City_type_CAT_B,
        City_type_CAT_C
    ]])

    # =====================================
    # SCALE INPUT FEATURES
    # =====================================

    scaled_features = scaler_X.transform(
        features
    )

    # =====================================
    # PREDICTION
    # =====================================

    prediction = model.predict(
        scaled_features
    )

    # =====================================
    # INVERSE TRANSFORM PREDICTION
    # =====================================

    final_prediction = scaler_y.inverse_transform(
        prediction.reshape(-1,1)
    )

    # =====================================
    # DISPLAY OUTPUT
    # =====================================

    st.success(
        f"🏡 Predicted House Price: ₹ {final_prediction[0][0]:,.2f}"
    )

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("About Project")

st.sidebar.write(
    """
    Machine Learning Project:
    
    - Algorithm: Support Vector Regression
    - Problem Type: Regression
    - Kernel Used: RBF
    - Frontend: Streamlit
    """
)

