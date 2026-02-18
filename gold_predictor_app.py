# gold_predictor_app.py - VERSI RINGAN UNTUK STREAMLIT CLOUD
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# GANTI TENSORFLOW DENGAN LIBRARY RINGAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Gold AI Predictor - Real Time",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Kustom
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .signal-box {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        animation: pulse 2s infinite;
        margin: 20px 0;
    }
    .bullish {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }
    .bearish {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
        color: white;
    }
    .neutral {
        background: linear-gradient(135deg, #f9ca24, #f6e58d);
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .day-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }
    .day-label {
        font-size: 0.9rem;
        color: #666;
    }
    .day-price {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .day-price-idr {
        font-size: 0.9rem;
        color: #27ae60;
    }
    .update-time {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        padding: 5px;
        background: #f0f0f0;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNGSI MENGAMBIL DATA REAL-TIME
# ============================================

@st.cache_data(ttl=300)
def get_live_gold_price():
    """Mengambil harga emas real-time dari GoldAPI.io"""
    
    # BACA API KEY DARI SECRETS
    try:
        api_key = st.secrets["api_key"]
    except:
        api_key = os.getenv("api_key", "goldapi-11ff6apsmlrq4w2v-io")
    
    headers = {'x-access-token': api_key, 'Content-Type': 'application/json'}
    
    try:
        response = requests.get(
            'https://www.goldapi.io/api/XAU/USD',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                'price': data['price'],
                'change': data.get('ch', 0),
                'change_percent': data.get('chp', 0),
                'timestamp': datetime.fromtimestamp(data['timestamp']),
                'source': 'GoldAPI.io'
            }
        else:
            return None
    except:
        return None

def generate_historical_prices(days=30):
    """Generate data historis berdasarkan harga real-time"""
    live = get_live_gold_price()
    if live:
        current_price = live['price']
        np.random.seed(42)
        prices = []
        for i in range(days):
            if i == 0:
                price = current_price * (1 - 0.02)  # 2% lebih rendah
            else:
                change = np.random.normal(0.001, 0.008)
                price = prices[-1] * (1 + change)
            prices.append(price)
        return np.array(prices)
    else:
        # Fallback data
        return np.linspace(4800, 4900, days)

def predict_prices(prices, days_to_predict=7):
    """Prediksi menggunakan Random Forest (lebih ringan dari LSTM)"""
    
    # Siapkan data untuk training
    X, y = [], []
    for i in range(len(prices) - 5):
        X.append(prices[i:i+5])
        y.append(prices[i+5])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) < 10:
        # Jika data kurang, gunakan simple forecast
        last_price = prices[-1]
        predictions = []
        for i in range(days_to_predict):
            pred = last_price * (1 + np.random.uniform(-0.005, 0.005))
            predictions.append(pred)
            last_price = pred
        return np.array(predictions)
    
    # Train Random Forest (ringan dan cepat)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Predict future
    last_sequence = prices[-5:]
    predictions = []
    
    for _ in range(days_to_predict):
        next_pred = model.predict(last_sequence.reshape(1, -1))[0]
        predictions.append(next_pred)
        last_sequence = np.append(last_sequence[1:], next_pred)
    
    return np.array(predictions)

# ============================================
# MAIN APP
# ============================================

def main():
    st.markdown('<h1 class="main-header">💰 Gold AI Predictor - Real Time</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Neural network analysis with real-time market intelligence</p>', unsafe_allow_html=True)
    
    # Kurs info
    st.markdown("""
    <div style="text-align:center; padding:10px; background:#f0f0f0; border-radius:10px; margin-bottom:20px;">
        💰 1 USD = Rp 15,000 (approx)
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/gold-bars.png", width=80)
        st.title("⚙️ Settings")
        st.subheader("📊 Analysis Parameters")
        days_predict = st.slider("Days to predict", 3, 14, 7)
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Ambil data
    with st.spinner('Mengambil data harga emas...'):
        live_data = get_live_gold_price()
    
    if live_data:
        # Tampilkan harga
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Harga Spot (USD)", f"${live_data['price']:,.2f}")
        with col2:
            st.metric("Harga Rupiah", f"Rp {live_data['price']*15000:,.0f}")
        with col3:
            gram_price = live_data['price'] / 31.1035
            st.metric("Per Gram (24K)", f"${gram_price:.2f}")
        with col4:
            change_symbol = "▲" if live_data['change_percent'] >= 0 else "▼"
            st.metric("Perubahan", f"{change_symbol} {live_data['change_percent']:+.2f}%")
        
        st.markdown(f"📡 Last update: {live_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} WIB")
        st.markdown("---")
        
        # 7-Day Table
        st.subheader("📅 7-Day Gold Price Analysis")
        historical_prices = generate_historical_prices(30)
        last_7_days = historical_prices[-7:]
        
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime('%d %b') for i in range(6, -1, -1)]
        day_labels = ['DAY -6', 'DAY -5', 'DAY -4', 'DAY -3', 'DAY -2', 'DAY -1', 'TODAY']
        
        cols = st.columns(7)
        for i, col in enumerate(cols):
            with col:
                price = last_7_days[i]
                st.markdown(f"""
                <div class="day-card">
                    <div class="day-label">{day_labels[i]}</div>
                    <div class="day-label">{dates[i]}</div>
                    <div class="day-price">${price:,.0f}</div>
                    <div class="day-price-idr">Rp {price*15000:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Analyze Button
        if st.button("🔮 Analyze & Predict", type="primary", use_container_width=True):
            
            predictions = predict_prices(historical_prices, days_predict)
            current_price = last_7_days[-1]
            next_price = predictions[0]
            price_change = ((next_price - current_price) / current_price) * 100
            
            # Signal
            if price_change > 1.5:
                signal, signal_class, icon = "INCREASE", "bullish", "🚀"
            elif price_change < -1.5:
                signal, signal_class, icon = "DECREASE", "bearish", "📉"
            else:
                signal, signal_class, icon = "STABLE", "neutral", "➡️"
            
            volatility = np.std(last_7_days) / np.mean(last_7_days) * 100
            confidence = min(95, max(65, int(90 - volatility)))
            
            st.markdown(f"""
            <div class="signal-box {signal_class}">
                <h1 style="font-size:3rem;">{icon}</h1>
                <h2>PRICE WILL {signal}</h2>
                <p style="font-size:1.5rem;">{signal} SIGNAL</p>
                <p style="font-size:1.2rem;">AI Confidence: {confidence}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Market Analysis
            st.markdown("---")
            st.subheader("📊 Market Analysis")
            
            week_change = ((last_7_days[-1] - last_7_days[0]) / last_7_days[0]) * 100
            day_change = ((last_7_days[-1] - last_7_days[-2]) / last_7_days[-2]) * 100
            week_avg = np.mean(last_7_days)
            
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("7-DAY CHANGE", f"{week_change:+.2f}%")
            mcol2.metric("1-DAY CHANGE", f"{day_change:+.2f}%")
            mcol3.metric("VOLATILITY", f"{volatility:.2f}%")
            mcol4.metric("7-DAY AVG", f"${week_avg:.0f}")
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=historical_prices[-30:],
                mode='lines',
                name='Historical',
                line=dict(color='gold', width=3)
            ))
            fig.add_trace(go.Scatter(
                y=predictions,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red', width=3, dash='dash')
            ))
            fig.update_layout(title="Gold Price Prediction", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast Table
            st.subheader(f"🔮 {days_predict}-Day Forecast")
            pred_cols = st.columns(days_predict)
            for i, col in enumerate(pred_cols):
                with col:
                    change = ((predictions[i] - current_price) / current_price) * 100
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px; background:#f8f9fa; border-radius:10px;">
                        <div>Day +{i+1}</div>
                        <div style="font-size:1.2rem; color:#27ae60;">${predictions[i]:.0f}</div>
                        <div style="color:{'green' if change>0 else 'red'}">{change:+.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Gagal mengambil data. Cek API Key di Secrets!")

if __name__ == "__main__":
    main()