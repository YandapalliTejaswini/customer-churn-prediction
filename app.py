import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Customer Churn AI",
    layout="wide"
)

# ------------------------------------------------
# DARK PREMIUM STYLE
# ------------------------------------------------
st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

section[data-testid="stSidebar"] {
background: linear-gradient(#141e30,#243b55);
}

.stButton>button {
background: linear-gradient(90deg,#ff416c,#ff4b2b);
color:white;
border-radius:10px;
font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = joblib.load("models/churn_pipeline.pkl")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
page = st.sidebar.radio(
    "📊 Navigation",
    ["Prediction", "Analytics", "Executive Summary"]
)

# =================================================
# PAGE 1 — PREDICTION
# =================================================
if page == "Prediction":

    st.title("🚀 Customer Churn Intelligence Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure", 0, 72, 12)

    with col2:
        monthly = st.number_input("Monthly Charges", 10.0, 150.0, 70.0)

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month","One year","Two year"]
        )

    col4, col5 = st.columns(2)

    with col4:
        internet = st.selectbox(
            "Internet Service",
            ["DSL","Fiber optic","No"]
        )

    with col5:
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check","Mailed check",
             "Bank transfer (automatic)",
             "Credit card (automatic)"]
        )

    # Input dataframe
    input_df = pd.DataFrame({
        "gender":["Male"],
        "SeniorCitizen":[0],
        "Partner":["Yes"],
        "Dependents":["No"],
        "tenure":[tenure],
        "PhoneService":["Yes"],
        "MultipleLines":["No"],
        "InternetService":[internet],
        "OnlineSecurity":["No"],
        "OnlineBackup":["No"],
        "DeviceProtection":["No"],
        "TechSupport":["No"],
        "StreamingTV":["No"],
        "StreamingMovies":["No"],
        "Contract":[contract],
        "PaperlessBilling":["Yes"],
        "PaymentMethod":[payment],
        "MonthlyCharges":[monthly],
        "TotalCharges":[monthly*tenure]
    })

    if st.button("🔮 Predict Customer Churn"):

        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error(f"⚠ HIGH CHURN RISK — {prob:.2%}")
        else:
            st.success(f"✅ CUSTOMER SAFE — {(1-prob):.2%}")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text':"Churn Risk (%)"},
            gauge={
                'axis':{'range':[0,100]},
                'steps':[
                    {'range':[0,30],'color':'green'},
                    {'range':[30,70],'color':'orange'},
                    {'range':[70,100],'color':'red'}
                ],
            }
        ))

        st.plotly_chart(fig,width=True)

        # Explanation
        st.subheader("🧠 AI Explanation")

        if contract == "Month-to-month":
            st.write("• Short-term contract increases churn risk")

        if monthly > 80:
            st.write("• High monthly charges detected")

        if tenure < 12:
            st.write("• New customers churn more often")

# =================================================
# PAGE 2 — ANALYTICS
# =================================================
elif page == "Analytics":

    st.title("📈 Customer Analytics")

    df = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

    with col2:
        st.subheader("Monthly Charges Trend")
        st.line_chart(df["MonthlyCharges"])

# =================================================
# PAGE 3 — EXECUTIVE SUMMARY
# =================================================
else:

    st.title("💼 Executive Business Summary")

    df = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    churn_rate = (df["Churn"]=="Yes").mean()*100
    avg_revenue = df["MonthlyCharges"].mean()*12

    c1, c2 = st.columns(2)

    c1.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
    c2.metric("Avg Annual Revenue", f"${avg_revenue:.0f}")

    st.success(
        "📌 Recommendation: Focus retention offers on month-to-month customers."
    )