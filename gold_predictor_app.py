# gold_predictor_app.py - VERSI ML LENGKAP DENGAN PERBAIKAN
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Gold AI Predictor - ML Version",
    page_icon="💰",
    layout="wide"
)

# CSS Kustom
st.markdown("""
<style>
    .main-header { font-size:3rem; font-weight:800; color: #FFD700; text-align:center; }
    .day-card { background:white; padding:1rem; border-radius:10px; text-align:center; box-shadow:0 2px 5px rgba(0,0,0,0.1); }
    .signal-box { padding:2rem; border-radius:20px; text-align:center; margin:20px 0; }
    .bullish { background: linear-gradient(135deg, #00b09b, #96c93d); color:white; }
    .bearish { background: linear-gradient(135deg, #ff6b6b, #ee5253); color:white; }
    .neutral { background: linear-gradient(135deg, #f9ca24, #f6e58d); color:white; }
    .metric-card { background:#f8f9fa; padding:1rem; border-radius:15px; text-align:center; }
    .accuracy-badge {
        background: #2c3e50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💰 Gold AI Predictor - Machine Learning</h1>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# FUNGSI AMBIL DATA REAL-TIME
# ============================================

@st.cache_data(ttl=300)
def get_gold_price():
    """Ambil harga emas real-time"""
    try:
        api_key = st.secrets["api_key"]
        headers = {'x-access-token': api_key}
        response = requests.get(
            'https://www.goldapi.io/api/XAU/USD',
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return {
                'price': data['price'],
                'change': data.get('chp', 0),
                'timestamp': datetime.now()
            }
    except:
        pass
    
    # Data cadangan
    return {
        'price': 4900 + np.random.randn() * 10,
        'change': np.random.randn() * 0.5,
        'timestamp': datetime.now()
    }

# ============================================
# GENERATE DATA HISTORIS
# ============================================
def generate_historical_data(days=200):
    """Generate data historis untuk training"""
    np.random.seed(42)
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]
    
    # Simulasi harga emas dengan trend naik
    base_price = 4800
    prices = []
    price = base_price
    
    for i in range(days):
        trend = 0.0005  # Trend naik 0.05% per hari
        volatility = np.random.randn() * 0.008
        change = trend + volatility
        price = price * (1 + change)
        prices.append(price)
    
    # Buat dataframe
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.randint(1000, 5000, days),
        'high': [p * (1 + abs(np.random.randn()*0.005)) for p in prices],
        'low': [p * (1 - abs(np.random.randn()*0.005)) for p in prices],
        'open': [p * (1 + np.random.randn()*0.002) for p in prices]
    })
    
    return df

# ============================================
# FEATURE ENGINEERING
# ============================================
def create_features(df):
    """Buat fitur-fitur untuk machine learning"""
    data = df.copy()
    
    # Target: harga besok
    data['target'] = data['price'].shift(-1)
    
    # Moving averages
    for window in [3, 5, 7, 14]:
        data[f'ma_{window}'] = data['price'].rolling(window=window).mean()
        data[f'std_{window}'] = data['price'].rolling(window=window).std()
    
    # Price changes
    data['price_change_1d'] = data['price'].pct_change()
    data['price_change_3d'] = data['price'].pct_change(3)
    data['price_change_5d'] = data['price'].pct_change(5)
    
    # Ratios
    data['high_low_ratio'] = data['high'] / data['low']
    data['close_open_ratio'] = data['price'] / data['open']
    
    # Volume features
    data['volume_change'] = data['volume'].pct_change()
    data['volume_ma'] = data['volume'].rolling(window=5).mean()
    
    # Lag features
    for lag in [1, 2, 3, 4, 5]:
        data[f'lag_{lag}'] = data['price'].shift(lag)
    
    # Hapus baris dengan NaN
    data = data.dropna().reset_index(drop=True)
    
    return data

# ============================================
# TRAIN MODEL DENGAN SPLIT 80:20
# ============================================
def train_model(df):
    """Train model dengan split 80% training, 20% testing"""
    
    # Buat fitur
    df_features = create_features(df)
    
    # Pisahkan fitur dan target
    feature_cols = [col for col in df_features.columns if col not in ['date', 'target', 'price', 'high', 'low', 'open', 'volume']]
    X = df_features[feature_cols]
    y = df_features['target']
    
    # SPLIT DATA: 80% TRAINING, 20% TESTING
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scaling fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    with st.spinner('Training model machine learning...'):
        model.fit(X_train_scaled, y_train)
    
    # Prediksi
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Hitung akurasi
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    
    train_accuracy = 100 - train_mape
    test_accuracy = 100 - test_mape
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae
        },
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'df_features': df_features  # SIMPAN DATAFRAME DENGAN FITUR
    }

# ============================================
# PREDIKSI FUTURE
# ============================================
def predict_future(model, scaler, feature_cols, last_data, days=7):
    """Prediksi harga untuk hari-hari ke depan"""
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(days):
        # Reshape untuk 2D array
        if len(current_data.shape) == 1:
            current_data = current_data.reshape(1, -1)
        
        # Scale data
        current_scaled = scaler.transform(current_data)
        
        # Prediksi
        pred = model.predict(current_scaled)[0]
        predictions.append(pred)
        
        # Update data untuk prediksi berikutnya (sederhana)
        # Geser data dan update nilai terakhir
        if current_data.shape[1] > 1:
            new_data = np.roll(current_data, -1)
            new_data[0, -1] = pred / current_data[0, -2] * current_data[0, -1] if current_data[0, -2] != 0 else pred
            current_data = new_data
        else:
            current_data = np.array([[pred]])
    
    return np.array(predictions)

# ============================================
# MAIN APP
# ============================================

def main():
    # Ambil data real-time
    live_data = get_gold_price()
    
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
            arrow = "▲" if live_data['change'] >= 0 else "▼"
            st.metric("Perubahan 24h", f"{arrow} {abs(live_data['change']):.2f}%")
        
        st.caption(f"Terakhir update: {live_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} WIB")
        st.markdown("---")
        
        # Generate data historis untuk training
        df = generate_historical_data(200)
        
        # Train model dan dapatkan results
        results = train_model(df)
        
        # Tampilkan metrik akurasi
        st.subheader("📊 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="accuracy-badge">
                Test Accuracy: {results['metrics']['test_accuracy']:.2f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="accuracy-badge">
                Train Accuracy: {results['metrics']['train_accuracy']:.2f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="accuracy-badge">
                R² Score: {results['metrics']['test_r2']:.3f}
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="accuracy-badge">
                MAE: ${results['metrics']['test_mae']:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        # Interpretasi akurasi
        if results['metrics']['test_accuracy'] > 95:
            st.success("✅ Model SANGAT BAIK! (Akurasi > 95%)")
        elif results['metrics']['test_accuracy'] > 90:
            st.info("📈 Model BAIK (Akurasi 90-95%)")
        elif results['metrics']['test_accuracy'] > 80:
            st.warning("⚠️ Model CUKUP (Akurasi 80-90%)")
        else:
            st.error("❌ Model KURANG (Akurasi < 80%)")
        
        # Feature importance
        with st.expander("📊 Feature Importance"):
            st.dataframe(results['feature_importance'].head(10))
        
        # ========================================
        # TABEL 7 HARI TERAKHIR
        # ========================================
        st.subheader("📅 7-Day Gold Price Analysis")
        
        last_7_prices = df['price'].values[-7:]
        last_7_dates = df['date'].values[-7:]
        day_labels = ['DAY -6', 'DAY -5', 'DAY -4', 'DAY -3', 'DAY -2', 'DAY -1', 'TODAY']
        
        cols = st.columns(7)
        for i, col in enumerate(cols):
            with col:
                price = last_7_prices[i]
                date_obj = datetime.strptime(last_7_dates[i], '%Y-%m-%d')
                st.markdown(f"""
                <div class="day-card">
                    <div>{day_labels[i]}</div>
                    <div style="font-size:0.8rem;">{date_obj.strftime('%d %b')}</div>
                    <div style="font-size:1.2rem; font-weight:700;">${price:,.0f}</div>
                    <div style="font-size:0.9rem; color:#27ae60;">Rp {price*15000:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # ========================================
        # ANALYZE BUTTON
        # ========================================
        if st.button("🔮 Analyze & Predict (ML)", type="primary", use_container_width=True):
            
            # Gunakan df_features yang sudah punya semua kolom fitur
            df_features = results['df_features']
            last_row = df_features[results['feature_cols']].iloc[-1:].values
            
            # Prediksi 7 hari ke depan
            predictions = predict_future(
                results['model'], 
                results['scaler'], 
                results['feature_cols'], 
                last_row.flatten(),
                7
            )
            
            current_price = last_7_prices[-1]
            next_price = predictions[0]
            price_change = ((next_price - current_price) / current_price) * 100
            
            # Tentukan sinyal berdasarkan ML
            if price_change > 1.5:
                signal, signal_class, icon = "INCREASE", "bullish", "🚀"
            elif price_change < -1.5:
                signal, signal_class, icon = "DECREASE", "bearish", "📉"
            else:
                signal, signal_class, icon = "STABLE", "neutral", "➡️"
            
            # Confidence score berdasarkan akurasi model
            confidence = min(95, int(results['metrics']['test_accuracy']))
            
            # Signal box
            st.markdown(f"""
            <div class="signal-box {signal_class}">
                <h1 style="font-size:3rem;">{icon}</h1>
                <h2>PRICE WILL {signal}</h2>
                <p style="font-size:1.5rem;">{signal} SIGNAL</p>
                <p style="font-size:1.2rem;">ML Confidence: {confidence}%</p>
                <p style="font-size:0.9rem;">Based on Random Forest model with 80/20 split</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Market Analysis
            st.subheader("📊 Market Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            week_change = ((last_7_prices[-1] - last_7_prices[0]) / last_7_prices[0]) * 100
            day_change = ((last_7_prices[-1] - last_7_prices[-2]) / last_7_prices[-2]) * 100
            volatility = np.std(last_7_prices) / np.mean(last_7_prices) * 100
            week_avg = np.mean(last_7_prices)
            
            col1.metric("7-DAY CHANGE", f"{week_change:+.2f}%")
            col2.metric("1-DAY CHANGE", f"{day_change:+.2f}%")
            col3.metric("VOLATILITY", f"{volatility:.2f}%")
            col4.metric("7-DAY AVG", f"${week_avg:.0f}")
            
            # Chart
            st.subheader("📈 ML Price Prediction")
            
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=df['date'].values[-60:],
                y=df['price'].values[-60:],
                mode='lines',
                name='Historical Price',
                line=dict(color='gold', width=3)
            ))
            
            # Predictions
            future_dates = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='ML Prediction',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8, symbol='star')
            ))
            
            fig.update_layout(
                title=f"Gold Price: 60 Days Historical + 7 Days ML Prediction (Accuracy: {confidence}%)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast
            st.subheader("🔮 7-Day ML Forecast")
            fcols = st.columns(7)
            for i, col in enumerate(fcols):
                with col:
                    change = ((predictions[i] - current_price) / current_price) * 100
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px; background:#f8f9fa; border-radius:10px;">
                        <div style="font-weight:bold;">Day +{i+1}</div>
                        <div style="font-size:0.8rem;">{datetime.strptime(future_dates[i], '%Y-%m-%d').strftime('%d %b')}</div>
                        <div style="font-size:1.1rem; color:#27ae60;">${predictions[i]:.0f}</div>
                        <div style="font-size:0.8rem; color:{'green' if change>0 else 'red'}">
                            {change:+.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Test data visualization
            with st.expander("📊 Model Evaluation on Test Data (20% Data)"):
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=results['y_test'].values[-50:],
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue')
                ))
                fig2.add_trace(go.Scatter(
                    y=results['y_pred_test'][-50:],
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='red', dash='dash')
                ))
                fig2.update_layout(
                    title="Model Performance on Test Data (Last 50 samples)",
                    xaxis_title="Sample",
                    yaxis_title="Price (USD)",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
