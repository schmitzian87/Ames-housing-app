import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Ames Housing Price Predictor", layout="wide")

st.title("🏠 Ames Housing Price Prediction Tool")
st.write(
    "This app predicts residential property prices in Ames, Iowa and highlights the key factors that influence home values."
)

# Load data
df = pd.read_csv("AmesHousing.csv")

# Basic cleaning
df = df.dropna(subset=["SalePrice"])

# Use strongest explanatory features from analysis
features = [
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Garage Area",
    "Total Bsmt SF",
    "1st Flr SF",
    "Year Built",
    "Full Bath",
    "Year Remod/Add",
    "TotRms AbvGrd",
    "Fireplaces"
]

model_df = df[features + ["SalePrice"]].copy()
model_df = model_df.fillna(model_df.mean(numeric_only=True))

X = model_df[features]
y = model_df["SalePrice"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Sidebar inputs
st.sidebar.header("Enter Home Features")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 6)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
garage_cars = st.sidebar.slider("Garage Capacity (Cars)", 0, 5, 2)
garage_area = st.sidebar.number_input("Garage Area (sq ft)", 0, 1500, 500)
total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
first_flr_sf = st.sidebar.number_input("First Floor Area (sq ft)", 300, 4000, 1000)
year_built = st.sidebar.number_input("Year Built", 1870, 2026, 2000)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 5, 2)
year_remod = st.sidebar.number_input("Year Remodeled/Added", 1950, 2026, 2005)
rooms = st.sidebar.slider("Total Rooms Above Ground", 2, 15, 6)
fireplaces = st.sidebar.slider("Fireplaces", 0, 4, 1)

input_data = pd.DataFrame({
    "Overall Qual": [overall_qual],
    "Gr Liv Area": [gr_liv_area],
    "Garage Cars": [garage_cars],
    "Garage Area": [garage_area],
    "Total Bsmt SF": [total_bsmt_sf],
    "1st Flr SF": [first_flr_sf],
    "Year Built": [year_built],
    "Full Bath": [full_bath],
    "Year Remod/Add": [year_remod],
    "TotRms AbvGrd": [rooms],
    "Fireplaces": [fireplaces]
})

prediction = model.predict(input_data)[0]

# Main dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Sale Price", f"${prediction:,.0f}")

with col2:
    st.metric("Model R²", f"{r2:.3f}")

with col3:
    st.metric("RMSE", f"${rmse:,.0f}")

st.divider()

# Prediction explanation
st.subheader("Prediction Interpretation")
st.write(
    f"Based on the home features entered, the estimated sale price is **${prediction:,.0f}**. "
    "This prediction should be used as decision support, not as an exact appraisal."
)

# Key insights
st.subheader("Key Insights from the Data")

insights = pd.DataFrame({
    "Feature": [
        "Overall Quality",
        "Above Ground Living Area",
        "Garage Capacity",
        "Garage Area",
        "Total Basement Area",
        "First Floor Area",
        "Year Built",
        "Full Bathrooms"
    ],
    "Correlation with Sale Price": [
        0.799,
        0.707,
        0.648,
        0.640,
        0.632,
        0.622,
        0.558,
        0.546
    ]
})

st.dataframe(insights, use_container_width=True)

st.bar_chart(insights.set_index("Feature"))

st.divider()

# Business value
st.subheader("Business Value for Decision Makers")

st.write("""
This tool helps buyers, sellers, and real estate analysts better understand what drives housing prices in Ames, Iowa.

**Main takeaways:**
- Higher overall quality is the strongest driver of price.
- Larger homes tend to sell for more.
- Garage size, basement space, bathrooms, and newer construction all add value.
- A non-technical user can adjust home features and immediately see how the predicted price changes.
""")

st.divider()

# Data preview
with st.expander("View Dataset Preview"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Model Features Used"):
    st.write(features)