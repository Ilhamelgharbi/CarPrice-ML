import streamlit as st
import requests
import os

# Streamlit page config
st.set_page_config(
    page_title="CarPriceML Predictor",
    page_icon="🚗",
    layout="centered"
)

# Header
st.markdown("<h1 style='text-align:center;color:#333;'> CarPriceML</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Predict Your Car's Price with AI</p>", unsafe_allow_html=True)

# Prediction form
st.markdown("### 📝 Enter Car Details")
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", ["Maruti", "Skoda", "Honda", "Hyundai", "Toyota"])
    year = st.number_input("Year", 2000, 2025, 2015, 1)
    fuel = st.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG"])
    km_driven = st.number_input("Kilometers Driven", 0, 500000, 50000, 1000)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7], index=2)

with col2:
    owner = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
    engine_cc = st.number_input("Engine CC", 500, 5000, 1200, 50)
    max_power_bhp = st.number_input("Max Power (BHP)", 30.0, 500.0, 80.0, 5.0)
    
    # Fixed Power per CC
    power_per_cc = round(max_power_bhp / engine_cc, 4) if engine_cc > 0 else 0.0667
    st.metric("Power per CC", f"{power_per_cc:.4f}")

# Submit button
submit = st.button("🔮 Predict Price")
API_URL = os.getenv("API_URL", "http://localhost:8000")

if submit:
    payload = {
        "max_power_bhp": float(max_power_bhp),
        "year": int(year),
        "engine_cc": int(engine_cc),
        "power_per_cc": float(power_per_cc),
        "seats": int(seats),
        "km_driven": int(km_driven),
        "brand": str(brand),
        "owner": str(owner),
        "fuel": str(fuel),
        "seller_type": str(seller_type),
        "transmission": str(transmission)
    }
    
    try:
        with st.spinner("🔄 Predicting..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                predicted_price = result['predicted_price_mad']
                
                # Main price display
                st.markdown(f"<h2 style='text-align:center;color:#059669;'>💰 Predicted Price: {predicted_price:,.2f} MAD</h2>", unsafe_allow_html=True)
                
                # Confidence metrics
                st.markdown("### 🎯 Confidence Metrics")
                conf_col1, conf_col2, conf_col3 = st.columns(3)
                
                with conf_col1:
                    st.metric("Confidence Score", f"{result.get('confidence_score', 0):.1f}%")
                with conf_col2:
                    st.metric("Std Deviation", f"{result.get('prediction_std', 0):,.0f} MAD")
                with conf_col3:
                    ci = result.get('confidence_interval', {})
                    ci_range = ci.get('upper', 0) - ci.get('lower', 0)
                    st.metric("CI Range", f"{ci_range:,.0f} MAD")
                
                # Show CI interval
                ci_lower = ci.get('lower', 0)
                ci_upper = ci.get('upper', 0)
                st.info(f"📊 95% Confidence Interval: {ci_lower:,.0f} - {ci_upper:,.0f} MAD")
                
                # Interpretation
                confidence_score = result.get('confidence_score', 0)
                if confidence_score >= 80:
                    st.success("🎯 Very Reliable - High confidence")
                elif confidence_score >= 60:
                    st.info("✅ Good Reliability")
                elif confidence_score >= 40:
                    st.warning("⚠️ Moderate Reliability")
                else:
                    st.error("❌ Lower Reliability")
            else:
                st.error(f"❌ Error {response.status_code}")
    except Exception as e:
        st.error(f"❌ {str(e)}")
