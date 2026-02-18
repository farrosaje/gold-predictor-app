# ai_lengkap.py
"""
PROGRAM AI LENGKAP DENGAN TENSORFLOW DAN KAGGLE
Fitur:
- Download dataset dari Kaggle
- Preprocessing data
- Membangun model Neural Network
- Training dan evaluasi
- Visualisasi hasil
- Prediksi dengan model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import subprocess
import sys
import time
from datetime import datetime

print("=" * 70)
print("PROGRAM AI LENGKAP DENGAN TENSORFLOW")
print("=" * 70)
print(f"Waktu mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}")
print("=" * 70)

# ============================================
# FUNGSI-FUNGSI UTILITY
# ============================================

def install_package(package):
    """Install package jika belum ada"""
    try:
        __import__(package)
        print(f"✓ {package} sudah terinstall")
    except ImportError:
        print(f"⏳ Menginstall {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_from_kaggle(dataset_name, destination="./data"):
    """Download dataset dari Kaggle"""
    print(f"\n📥 Downloading dataset: {dataset_name}")
    
    # Buat folder
    Path(destination).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download menggunakan kaggle API
        result = subprocess.run([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", dataset_name,
            "-p", destination
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        
        # Extract zip files
        zip_files = list(Path(destination).glob("*.zip"))
        if zip_files:
            for zip_file in zip_files:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(destination)
                print(f"📦 Extracted: {zip_file.name}")
        
        print(f"✅ Dataset downloaded to {destination}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def plot_training_history(history, save_path=None):
    """Plot history training"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2, color='blue')
    if 'val_accuracy' in history.history:
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='orange')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2, color='blue')
    if 'val_loss' in history.history:
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='orange')
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate (jika ada)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], linewidth=2, color='green')
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Precision/Recall (jika binary classification dan metrics tersedia)
    if 'precision' in history.history:
        axes[1, 1].plot(history.history['precision'], label='Precision', linewidth=2)
        axes[1, 1].plot(history.history['recall'], label='Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Hapus subplot yang tidak terpakai
        fig.delaxes(axes[1, 1])
        if 'lr' not in history.history:
            fig.delaxes(axes[1, 0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()

# ============================================
# KELAS UNTUK DATA PROCESSING
# ============================================

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_csv(self, filename):
        """Load CSV file"""
        file_path = os.path.join(self.data_path, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"\n📄 File: {filename}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()[:10]}...")
            return df
        else:
            print(f"❌ File not found: {file_path}")
            return None
    
    def preprocess_classification(self, df, target_column, test_size=0.2):
        """Preprocess untuk klasifikasi"""
        print("\n🔄 Preprocessing data untuk klasifikasi...")
        
        # Pisahkan fitur dan target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            print(f"❌ Target column '{target_column}' not found!")
            return None
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print(f"⚠️ Found {X.isnull().sum().sum()} missing values. Filling with mean...")
            X = X.fillna(X.mean())
        
        # Encode target jika string
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            print(f"Classes: {self.label_encoder.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Normalisasi
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Training data: {X_train_scaled.shape}")
        print(f"✅ Testing data: {X_test_scaled.shape}")
        print(f"✅ Number of classes: {len(np.unique(y))}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'num_classes': len(np.unique(y)),
            'label_encoder': self.label_encoder
        }

# ============================================
# MEMBANGUN MODEL
# ============================================

def build_model(input_shape, num_classes, model_type='basic'):
    """Membangun model berdasarkan tipe"""
    
    if model_type == 'basic':
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
        ])
    
    elif model_type == 'deep':
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
        ])
    
    else:  # simple
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
        ])
    
    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics
    )
    
    return model

# ============================================
# FUNGSI UTAMA
# ============================================

def main():
    """Fungsi utama program"""
    
    print("\n" + "=" * 70)
    print("MEMULAI PROGRAM UTAMA")
    print("=" * 70)
    
    # Pilih mode
    print("\nPilih mode:")
    print("1. Menggunakan dataset dari Kaggle")
    print("2. Menggunakan data dummy (testing)")
    print("3. Load model yang sudah ada")
    
    mode = input("\nMasukkan pilihan (1/2/3): ").strip()
    
    # ========== MODE 1: KAGGLE DATASET ==========
    if mode == '1':
        print("\n" + "-" * 40)
        print("MODE: DATASET KAGGLE")
        print("-" * 40)
        
        # Dataset yang tersedia
        datasets = {
            '1': ('uciml/iris', 'Iris.csv', 'Species', 'Dataset Bunga Iris (Klasifikasi)'),
            '2': ('puneet6060/intel-image-classification', None, None, 'Dataset Gambar Intel (Image Classification)'),
            '3': ('uciml/breast-cancer-wisconsin-data', 'data.csv', 'diagnosis', 'Dataset Kanker Payudara'),
            '4': ('rashikrahmanpritom/heart-attack-analysis-prediction-dataset', 'heart.csv', 'output', 'Dataset Serangan Jantung'),
        }
        
        print("\nPilih dataset:")
        for key, (_, _, _, desc) in datasets.items():
            print(f"{key}. {desc}")
        print("5. Custom dataset (masukkan nama sendiri)")
        
        dataset_choice = input("\nPilih dataset (1-5): ").strip()
        
        if dataset_choice == '5':
            dataset_name = input("Masukkan nama dataset Kaggle (format: username/nama-dataset): ").strip()
            csv_name = input("Masukkan nama file CSV (jika ada): ").strip()
            target_col = input("Masukkan nama kolom target: ").strip()
        elif dataset_choice in datasets:
            dataset_name, csv_name, target_col, _ = datasets[dataset_choice]
        else:
            print("Pilihan tidak valid, menggunakan dataset default: Iris")
            dataset_name, csv_name, target_col, _ = datasets['1']
        
        # Download dataset
        success = download_from_kaggle(dataset_name)
        
        if not success:
            print("❌ Gagal download dataset. Beralih ke data dummy...")
            mode = '2'
        else:
            # Cari file CSV
            data_dir = './data'
            csv_files = list(Path(data_dir).glob('*.csv'))
            
            if csv_files:
                # Gunakan file CSV pertama jika tidak ditentukan
                if not csv_name:
                    csv_name = csv_files[0].name
                    print(f"📄 Menggunakan file: {csv_name}")
                
                # Load dan proses data
                processor = DataProcessor(data_dir)
                df = processor.load_csv(csv_name)
                
                if df is not None and target_col:
                    data = processor.preprocess_classification(df, target_col)
                    
                    if data:
                        X_train, X_test = data['X_train'], data['X_test']
                        y_train, y_test = data['y_train'], data['y_test']
                        num_classes = data['num_classes']
                        input_shape = X_train.shape[1]
                        
                        print(f"\n🧠 Membangun model dengan {input_shape} fitur dan {num_classes} kelas...")
                        model = build_model(input_shape, num_classes, 'basic')
                        model.summary()
                        
                        # Callbacks
                        callbacks = [
                            keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=15,
                                restore_best_weights=True
                            ),
                            keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.5,
                                patience=5,
                                min_lr=1e-6
                            ),
                            keras.callbacks.ModelCheckpoint(
                                'best_model.h5',
                                monitor='val_accuracy',
                                save_best_only=True
                            )
                        ]
                        
                        # Training
                        print("\n🎯 Melatih model...")
                        history = model.fit(
                            X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=100,
                            batch_size=16,
                            callbacks=callbacks,
                            verbose=1
                        )
                        
                        # Evaluasi
                        print("\n📈 Evaluasi model:")
                        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)[:2]
                        print(f"Test Loss: {loss:.4f}")
                        print(f"Test Accuracy: {accuracy:.4f}")
                        
                        # Plot
                        plot_training_history(history, 'training_history.png')
                        
                        # Simpan model
                        model.save('kaggle_model.h5')
                        print("💾 Model saved as 'kaggle_model.h5'")
                        
                        # Prediksi sample
                        print("\n🔮 Sample predictions:")
                        sample_idx = np.random.choice(len(X_test), 5, replace=False)
                        sample_data = X_test[sample_idx]
                        sample_true = y_test[sample_idx]
                        
                        predictions = model.predict(sample_data)
                        
                        if num_classes == 2:
                            pred_classes = (predictions > 0.5).astype(int).flatten()
                            for i, (pred, true) in enumerate(zip(pred_classes, sample_true)):
                                print(f"Sample {i+1}: Predicted={pred}, Actual={true}, Prob={predictions[i][0]:.4f}")
                        else:
                            pred_classes = np.argmax(predictions, axis=1)
                            for i, (pred, true) in enumerate(zip(pred_classes, sample_true)):
                                confidence = np.max(predictions[i])
                                print(f"Sample {i+1}: Predicted={pred}, Actual={true}, Confidence={confidence:.4f}")
                        
                        print("\n✅ Training selesai!")
            
    # ========== MODE 2: DATA DUMMY ==========
    if mode == '2':
        print("\n" + "-" * 40)
        print("MODE: DATA DUMMY")
        print("-" * 40)
        
        # Generate data dummy
        print("\n📊 Generating dummy data...")
        n_samples = 5000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        print(f"Generated {n_samples} samples with {n_features} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build model
        print("\n🧠 Building model...")
        model = build_model(n_features, 2, 'deep')
        model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.TensorBoard(log_dir='./logs')
        ]
        
        # Training
        print("\n🎯 Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluasi
        print("\n📈 Evaluating model...")
        loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        
        # Plot
        plot_training_history(history, 'dummy_training_history.png')
        
        # Simpan model
        model.save('dummy_model.h5')
        print("💾 Model saved as 'dummy_model.h5'")
        
        print("\n✅ Training selesai!")
    
    # ========== MODE 3: LOAD MODEL ==========
    elif mode == '3':
        print("\n" + "-" * 40)
        print("MODE: LOAD MODEL")
        print("-" * 40)
        
        # Cari file model
        model_files = list(Path('.').glob('*.h5'))
        
        if model_files:
            print("\nModel files found:")
            for i, f in enumerate(model_files):
                size = os.path.getsize(f) / (1024 * 1024)  # Convert to MB
                print(f"{i+1}. {f.name} ({size:.2f} MB)")
            
            choice = input("\nPilih model (nomor): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(model_files):
                    model_path = model_files[idx]
                    print(f"📂 Loading model: {model_path}")
                    
                    model = keras.models.load_model(model_path)
                    model.summary()
                    
                    print("\n✅ Model loaded successfully!")
                    
                    # Test dengan data random
                    print("\n🔮 Testing with random data...")
                    input_shape = model.input_shape[1:]
                    
                    if len(input_shape) == 1:  # Tabular data
                        test_data = np.random.randn(5, input_shape[0])
                        predictions = model.predict(test_data)
                        
                        print("Predictions:")
                        for i, pred in enumerate(predictions):
                            if pred.shape[0] == 1:  # Binary
                                print(f"Sample {i+1}: {pred[0]:.4f} -> {'Positive' if pred[0] > 0.5 else 'Negative'}")
                            else:  # Multi-class
                                class_pred = np.argmax(pred)
                                conf = np.max(pred)
                                print(f"Sample {i+1}: Class {class_pred} (Confidence: {conf:.4f})")
                else:
                    print("❌ Invalid choice")
            except ValueError:
                print("❌ Invalid input")
        else:
            print("❌ No model files found (.h5)")
            print("Train a model first (mode 1 or 2)")

# ============================================
# JALANKAN PROGRAM
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 70)
        print(f"Waktu selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print("\nTerima kasih telah menggunakan program ini!")
        input("Press Enter to exit...")