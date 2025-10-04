# Chapter 4: Results and Analysis - AI-Enhanced IDS in Cloud Computing
# Comprehensive Implementation and Evaluation - COMPLETE VERSION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LSTM, Conv1D, MaxPooling1D, 
                                   Dropout, BatchNormalization, Input, 
                                   Concatenate, Flatten, Reshape)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Imbalanced Learning
from imblearn.over_sampling import SMOTE

# Explainability
import shap

# Additional Libraries
import time
import pickle
from collections import Counter
import itertools

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class IDSDataPreprocessor:
    """Comprehensive data preprocessing for IDS datasets"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        
    def load_and_preprocess_data(self, dataset_name='CICIDS2017'):
        """Load and preprocess IDS datasets"""
        print(f"Loading {dataset_name} dataset...")
        
        # Simulated data loading - replace with actual dataset paths
        if dataset_name == 'CICIDS2017':
            # Load CICIDS2017 dataset
            data = self._simulate_cicids2017_data()
        elif dataset_name == 'CICDDoS2019':
            data = self._simulate_cicddos2019_data()
        elif dataset_name == 'UNSW-NB15':
            data = self._simulate_unsw_nb15_data()
        
        return data
    
    def _simulate_cicids2017_data(self):
        """Simulate CICIDS2017 dataset structure"""
        np.random.seed(42)
        n_samples = 50000
        n_features = 78
        
        # Generate synthetic network traffic features
        feature_names = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
            'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
            'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
            'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
            'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
            'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std',
            'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ] + [f'Feature_{i}' for i in range(68, n_features)]
        
        # Generate feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Create realistic attack patterns
        attack_types = ['BENIGN', 'DoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack', 'DDoS']
        attack_probabilities = [0.7, 0.1, 0.05, 0.05, 0.02, 0.05, 0.03]
        
        y = np.random.choice(attack_types, n_samples, p=attack_probabilities)
        
        # Add realistic correlations for different attack types
        for i, attack in enumerate(attack_types[1:], 1):
            mask = y == attack
            if attack in ['DoS', 'DDoS']:
                X[mask, :5] *= np.random.uniform(2, 5, (np.sum(mask), 5))  # High packet rates
            elif attack == 'PortScan':
                X[mask, 5:10] *= np.random.uniform(0.1, 0.5, (np.sum(mask), 5))  # Low packet sizes
            elif attack == 'Bot':
                X[mask, 10:15] *= np.random.uniform(1.5, 3, (np.sum(mask), 5))  # Moderate anomalies
        
        data = pd.DataFrame(X, columns=feature_names[:n_features])
        data['Label'] = y
        
        return data
    
    def _simulate_cicddos2019_data(self):
        """Simulate CICDDoS2019 dataset - CORRECTED VERSION"""
        np.random.seed(43)
        n_samples = 30000
        n_features = 88
        
        X = np.random.randn(n_samples, n_features)
        ddos_types = ['BENIGN', 'UDP-lag', 'TFTP', 'LDAP', 'MSSQL', 'NetBIOS', 'SNMP', 'SSDP', 'DNS', 'NTP']
        
        # Fixed probabilities that sum to exactly 1.0
        probabilities = [0.5, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055]
        # Ensure they sum to 1.0
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        y = np.random.choice(ddos_types, n_samples, p=probabilities)
        
        feature_names = [f'DDoS_Feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data['Label'] = y
        
        return data
    
    def _simulate_unsw_nb15_data(self):
        """Simulate UNSW-NB15 dataset - CORRECTED VERSION"""
        np.random.seed(44)
        n_samples = 40000
        n_features = 49
        
        X = np.random.randn(n_samples, n_features)
        attack_categories = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 
                           'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms']
        
        # Fixed probabilities that sum to exactly 1.0
        probabilities = [0.6, 0.18, 0.04, 0.02, 0.04, 0.06, 0.02, 0.02, 0.01, 0.01]
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        y = np.random.choice(attack_categories, n_samples, p=probabilities)
        
        feature_names = [f'UNSW_Feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data['Label'] = y
        
        return data
    
    def preprocess_dataset(self, data, target_col='Label'):
        """Comprehensive preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Separate features and target
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Feature selection using correlation matrix and RFE
        correlation_matrix = X_scaled.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        
        # Remove highly correlated features (threshold > 0.95)
        high_corr_features = [column for column in upper_triangle.columns 
                            if any(upper_triangle[column] > 0.95)]
        X_scaled = X_scaled.drop(high_corr_features, axis=1)
        
        # Select best features using SelectKBest
        k_best = min(50, X_scaled.shape[1])  # Select top 50 features or all if less
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = selector.fit_transform(X_scaled, y_encoded)
        selected_features = X_scaled.columns[selector.get_support()].tolist()
        
        X_final = pd.DataFrame(X_selected, columns=selected_features)
        
        print(f"Preprocessing complete. Final shape: {X_final.shape}")
        print(f"Class distribution: {Counter(y)}")
        
        return X_final, y_encoded, y

class IDSModelSuite:
    """Comprehensive suite of IDS models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.training_history = {}
        
    def build_random_forest(self, n_estimators=100, random_state=42):
        """Build Random Forest model"""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.models['Random Forest'] = model
        return model
    
    def build_svm(self, kernel='rbf', random_state=42):
        """Build SVM model"""
        model = SVC(
            kernel=kernel,
            random_state=random_state,
            class_weight='balanced',
            probability=True
        )
        self.models['SVM'] = model
        return model
    
    def build_lstm(self, input_shape, num_classes):
        """Build LSTM model for sequential pattern recognition"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['LSTM'] = model
        return model
    
    def build_cnn(self, input_shape, num_classes):
        """Build CNN model for spatial pattern recognition"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            Flatten(),
            Dense(100, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['CNN'] = model
        return model
    
    def build_autoencoder(self, input_dim, encoding_dim=32):
        """Build Autoencoder for anomaly detection"""
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.models['Autoencoder'] = autoencoder
        self.models['Encoder'] = encoder
        return autoencoder, encoder
    
    def build_hybrid_cnn_lstm(self, input_shape, num_classes):
        """Build hybrid CNN-LSTM model"""
        # CNN branch
        cnn_input = Input(shape=input_shape)
        cnn_conv1 = Conv1D(64, 3, activation='relu')(cnn_input)
        cnn_pool1 = MaxPooling1D(2)(cnn_conv1)
        cnn_conv2 = Conv1D(32, 3, activation='relu')(cnn_pool1)
        cnn_pool2 = MaxPooling1D(2)(cnn_conv2)
        
        # LSTM branch
        lstm_input = Input(shape=input_shape)
        lstm_layer1 = LSTM(64, return_sequences=True)(lstm_input)
        lstm_dropout1 = Dropout(0.2)(lstm_layer1)
        lstm_layer2 = LSTM(32, return_sequences=False)(lstm_dropout1)
        lstm_dropout2 = Dropout(0.2)(lstm_layer2)
        
        # Flatten CNN output
        cnn_flatten = Flatten()(cnn_pool2)
        
        # Merge branches
        merged = Concatenate()([cnn_flatten, lstm_dropout2])
        dense1 = Dense(100, activation='relu')(merged)
        batch_norm = BatchNormalization()(dense1)
        dropout = Dropout(0.5)(batch_norm)
        dense2 = Dense(50, activation='relu')(dropout)
        output = Dense(num_classes, activation='softmax')(dense2)
        
        model = Model(inputs=[cnn_input, lstm_input], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['Hybrid CNN-LSTM'] = model
        return model

class IDSEvaluator:
    """Comprehensive evaluation framework for IDS models"""
    
    def __init__(self):
        self.results = {}
        self.confusion_matrices = {}
        self.roc_curves = {}
        
    def evaluate_model(self, model, model_name, X_test, y_test, y_test_categorical=None):
        """Comprehensive model evaluation"""
        print(f"\nEvaluating {model_name}...")
        
        start_time = time.time()
        
        if model_name in ['LSTM', 'CNN', 'Hybrid CNN-LSTM']:
            # Deep learning model evaluation
            if model_name == 'Hybrid CNN-LSTM':
                predictions = model.predict([X_test, X_test])
            else:
                predictions = model.predict(X_test)
            
            y_pred = np.argmax(predictions, axis=1)
            y_pred_proba = predictions
            y_true = np.argmax(y_test_categorical, axis=1) if y_test_categorical is not None else y_test
            
        else:
            # Traditional ML model evaluation
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            y_true = y_test
        
        detection_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC for multiclass
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) > 2:
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                roc_auc = 0.0
        else:
            roc_auc = 0.0
        
        # Store results
        self.results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Detection_Time': detection_time / len(X_test),  # Average time per sample
            'Predictions': y_pred,
            'True_Labels': y_true
        }
        
        # Store confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices[model_name] = cm
        
        print(f"{model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Avg Detection Time: {detection_time/len(X_test)*1000:.2f} ms")
        
        return self.results[model_name]
    
    def cross_validate_model(self, model, model_name, X, y, cv=5):
        """Perform k-fold cross-validation"""
        print(f"\nPerforming {cv}-fold cross-validation for {model_name}...")
        
        if model_name not in ['LSTM', 'CNN', 'Hybrid CNN-LSTM', 'Autoencoder']:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted', n_jobs=-1)
            cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall_weighted', n_jobs=-1)
            cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
            
            cv_results = {
                'CV_Accuracy_Mean': cv_scores.mean(),
                'CV_Accuracy_Std': cv_scores.std(),
                'CV_Precision_Mean': cv_precision.mean(),
                'CV_Precision_Std': cv_precision.std(),
                'CV_Recall_Mean': cv_recall.mean(),
                'CV_Recall_Std': cv_recall.std(),
                'CV_F1_Mean': cv_f1.mean(),
                'CV_F1_Std': cv_f1.std()
            }
            
            print(f"Cross-validation results for {model_name}:")
            print(f"  Accuracy: {cv_results['CV_Accuracy_Mean']:.4f} (+/- {cv_results['CV_Accuracy_Std']*2:.4f})")
            print(f"  Precision: {cv_results['CV_Precision_Mean']:.4f} (+/- {cv_results['CV_Precision_Std']*2:.4f})")
            print(f"  Recall: {cv_results['CV_Recall_Mean']:.4f} (+/- {cv_results['CV_Recall_Std']*2:.4f})")
            print(f"  F1-Score: {cv_results['CV_F1_Mean']:.4f} (+/- {cv_results['CV_F1_Std']*2:.4f})")
            
            return cv_results
        else:
            print(f"Cross-validation not performed for deep learning model: {model_name}")
            return None
    
    def plot_confusion_matrices(self, class_names=None):
        """Plot confusion matrices for all models"""
        n_models = len(self.confusion_matrices)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (model_name, cm) in enumerate(self.confusion_matrices.items()):
            if idx >= 6:  # Limit to 6 models for display
                break
                
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=class_names[:cm.shape[1]] if class_names else None,
                       yticklabels=class_names[:cm.shape[0]] if class_names else None)
            axes[idx].set_title(f'{model_name} Confusion Matrix')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
        
        # Hide unused subplots
        for idx in range(n_models, 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self):
        """Plot performance comparison across models"""
        if not self.results:
            return
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        models = list(self.results.keys())
        
        # Create performance matrix
        performance_matrix = np.zeros((len(models), len(metrics)))
        
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics):
                performance_matrix[i, j] = self.results[model][metric]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(performance_matrix, annot=True, fmt='.3f', 
                   xticklabels=metrics, yticklabels=models,
                   cmap='viridis', center=0.5)
        plt.title('Model Performance Comparison')
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Bar plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            axes[idx].bar(models, values, color='skyblue', edgecolor='navy')
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_ylabel(metric)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Detection time comparison
        detection_times = [self.results[model]['Detection_Time'] * 1000 for model in models]  # Convert to ms
        axes[5].bar(models, detection_times, color='lightcoral', edgecolor='darkred')
        axes[5].set_title('Average Detection Time Comparison')
        axes[5].set_ylabel('Time (ms)')
        axes[5].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(detection_times):
            axes[5].text(i, v + max(detection_times) * 0.01, f'{v:.2f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('detailed_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

class ComprehensiveIDSAnalysis:
    """Main class orchestrating the complete IDS analysis"""
    
    def __init__(self):
        self.preprocessor = IDSDataPreprocessor()
        self.model_suite = IDSModelSuite()
        self.evaluator = IDSEvaluator()
        
        self.datasets = {}
        self.processed_data = {}
        self.trained_models = {}
        
    def run_comprehensive_analysis(self):
        """Execute complete IDS analysis pipeline"""
        print("="*80)
        print("CHAPTER 4: RESULTS AND ANALYSIS")
        print("AI-Enhanced Intrusion Detection Systems in Cloud Computing")
        print("="*80)
        
        # Step 1: Data Loading and Preprocessing
        self.load_and_preprocess_datasets()
        
        # Step 2: Model Development and Training
        self.train_all_models()
        
        # Step 3: Model Evaluation
        self.evaluate_all_models()
        
        # Step 4: Generate Comprehensive Report
        self.generate_analysis_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - All results saved to files")
        print("="*80)
    
    def load_and_preprocess_datasets(self):
        """Load and preprocess all datasets"""
        print("\n4.1 DATA PREPROCESSING AND ANALYSIS")
        print("-" * 50)
        
        dataset_names = ['CICIDS2017']  # Focus on primary dataset for this example
        
        for dataset_name in dataset_names:
            print(f"\nProcessing {dataset_name} dataset...")
            
            # Load raw data
            raw_data = self.preprocessor.load_and_preprocess_data(dataset_name)
            self.datasets[dataset_name] = raw_data
            
            # Preprocess
            X, y_encoded, y_original = self.preprocessor.preprocess_dataset(raw_data)
            
            # Apply SMOTE for class balance
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y_encoded)
            
            print(f"Original shape: {X.shape}, Balanced shape: {X_balanced.shape}")
            print(f"Original class distribution: {Counter(y_encoded)}")
            print(f"Balanced class distribution: {Counter(y_balanced)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
            
            self.processed_data[dataset_name] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': X.columns.tolist(),
                'class_names': self.preprocessor.label_encoder.classes_.tolist(),
                'num_classes': len(np.unique(y_balanced)),
                'y_original': y_original
            }
        
        # Use CICIDS2017 as primary dataset for detailed analysis
        self.primary_dataset = 'CICIDS2017'
        print(f"\nUsing {self.primary_dataset} as primary dataset for detailed analysis")
    
    def train_all_models(self):
        """Train all models on the primary dataset"""
        print(f"\n4.2 MODEL DEVELOPMENT AND TRAINING")
        print("-" * 50)
        
        data = self.processed_data[self.primary_dataset]
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        num_classes = data['num_classes']
        
        print(f"Training models on {self.primary_dataset} dataset")
        print(f"Training set shape: {X_train.shape}")
        print(f"Number of classes: {num_classes}")
        
        # Convert to numpy arrays for consistency
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Prepare data for deep learning models
        y_train_categorical = to_categorical(y_train, num_classes=num_classes)
        y_test_categorical = to_categorical(y_test, num_classes=num_classes)
        
        # Store categorical labels for later use
        data['y_train_categorical'] = y_train_categorical
        data['y_test_categorical'] = y_test_categorical
        
        # 1. Train Random Forest
        print("\n4.2.1 Training Random Forest...")
        rf_model = self.model_suite.build_random_forest(n_estimators=100)
        rf_model.fit(X_train_np, y_train)
        self.trained_models['Random Forest'] = rf_model
        
        # 2. Train SVM
        print("\n4.2.2 Training Support Vector Machine...")
        svm_model = self.model_suite.build_svm(kernel='rbf')
        svm_model.fit(X_train_np, y_train)
        self.trained_models['SVM'] = svm_model
        
        # 3. Train LSTM
        print("\n4.2.3 Training LSTM Network...")
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = X_train_np.reshape(X_train_np.shape[0], 1, X_train_np.shape[1])
        X_test_lstm = X_test_np.reshape(X_test_np.shape[0], 1, X_test_np.shape[1])
        
        lstm_model = self.model_suite.build_lstm(
            input_shape=(1, X_train_np.shape[1]), 
            num_classes=num_classes
        )
        
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        # Train LSTM
        lstm_history = lstm_model.fit(
            X_train_lstm, y_train_categorical,
            validation_data=(X_test_lstm, y_test_categorical),
            epochs=50,
            batch_size=128,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.trained_models['LSTM'] = lstm_model
        self.model_suite.training_history['LSTM'] = lstm_history
        
        # Store reshaped data for evaluation
        data['X_test_lstm'] = X_test_lstm
        
        # 4. Train CNN
        print("\n4.2.4 Training CNN...")
        # Reshape for CNN (samples, features, 1)
        X_train_cnn = X_train_np.reshape(X_train_np.shape[0], X_train_np.shape[1], 1)
        X_test_cnn = X_test_np.reshape(X_test_np.shape[0], X_test_np.shape[1], 1)
        
        cnn_model = self.model_suite.build_cnn(
            input_shape=(X_train_np.shape[1], 1), 
            num_classes=num_classes
        )
        
        # Train CNN
        cnn_history = cnn_model.fit(
            X_train_cnn, y_train_categorical,
            validation_data=(X_test_cnn, y_test_categorical),
            epochs=50,
            batch_size=128,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.trained_models['CNN'] = cnn_model
        self.model_suite.training_history['CNN'] = cnn_history
        
        # Store reshaped data for evaluation
        data['X_test_cnn'] = X_test_cnn
        
        # 5. Train Autoencoder (for anomaly detection)
        print("\n4.2.5 Training Autoencoder...")
        autoencoder, encoder = self.model_suite.build_autoencoder(
            input_dim=X_train_np.shape[1], 
            encoding_dim=32
        )
        
        # Train on normal traffic only (assuming class 0 is BENIGN)
        normal_mask = y_train == 0
        X_train_normal = X_train_np[normal_mask]
        
        autoencoder_history = autoencoder.fit(
            X_train_normal, X_train_normal,
            epochs=50,
            batch_size=128,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.trained_models['Autoencoder'] = autoencoder
        self.trained_models['Encoder'] = encoder
        self.model_suite.training_history['Autoencoder'] = autoencoder_history
        
        # 6. Train Hybrid CNN-LSTM
        print("\n4.2.6 Training Hybrid CNN-LSTM...")
        hybrid_model = self.model_suite.build_hybrid_cnn_lstm(
            input_shape=(X_train_np.shape[1], 1), 
            num_classes=num_classes
        )
        
        # Train hybrid model (uses same input for both branches)
        hybrid_history = hybrid_model.fit(
            [X_train_cnn, X_train_cnn], y_train_categorical,
            validation_data=([X_test_cnn, X_test_cnn], y_test_categorical),
            epochs=50,
            batch_size=128,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.trained_models['Hybrid CNN-LSTM'] = hybrid_model
        self.model_suite.training_history['Hybrid CNN-LSTM'] = hybrid_history
        
        print(f"\nAll models trained successfully!")
        print(f"Total models: {len(self.trained_models)}")
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print(f"\n4.3 MODEL EVALUATION AND PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        data = self.processed_data[self.primary_dataset]
        X_test = data['X_test']
        y_test = data['y_test']
        y_test_categorical = data['y_test_categorical']
        class_names = data['class_names']
        
        # Convert to numpy
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Evaluate each model
        for model_name, model in self.trained_models.items():
            if model_name == 'Encoder':  # Skip encoder (part of autoencoder)
                continue
                
            if model_name == 'LSTM':
                self.evaluator.evaluate_model(
                    model, model_name, data['X_test_lstm'], y_test, y_test_categorical
                )
            elif model_name == 'CNN':
                self.evaluator.evaluate_model(
                    model, model_name, data['X_test_cnn'], y_test, y_test_categorical
                )
            elif model_name == 'Hybrid CNN-LSTM':
                self.evaluator.evaluate_model(
                    model, model_name, [data['X_test_cnn'], data['X_test_cnn']], y_test, y_test_categorical
                )
            elif model_name == 'Autoencoder':
                # Special handling for autoencoder (anomaly detection)
                self._evaluate_autoencoder(model, X_test_np, y_test)
            else:
                self.evaluator.evaluate_model(
                    model, model_name, X_test_np, y_test
                )
        
        # Cross-validation for traditional ML models
        print(f"\n4.3.1 Cross-Validation Results")
        print("-" * 30)
        
        X_train_np = data['X_train'].values if hasattr(data['X_train'], 'values') else data['X_train']
        y_train = data['y_train']
        
        for model_name in ['Random Forest', 'SVM']:
            if model_name in self.trained_models:
                self.evaluator.cross_validate_model(
                    self.trained_models[model_name], model_name, X_train_np, y_train
                )
        
        # Generate visualizations
        print(f"\n4.3.2 Generating Performance Visualizations...")
        self.evaluator.plot_confusion_matrices(class_names)
        self.evaluator.plot_performance_comparison()
        
        # Plot training histories for deep learning models
        self._plot_training_histories()
    
    def _evaluate_autoencoder(self, autoencoder, X_test, y_test):
        """Special evaluation for autoencoder-based anomaly detection"""
        print(f"\nEvaluating Autoencoder (Anomaly Detection)...")
        
        start_time = time.time()
        
        # Reconstruct test data
        X_test_pred = autoencoder.predict(X_test)
        
        # Calculate reconstruction error
        mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
        
        # Determine threshold (95th percentile of normal traffic reconstruction error)
        normal_mask = y_test == 0
        if np.sum(normal_mask) > 0:
            threshold = np.percentile(mse[normal_mask], 95)
        else:
            threshold = np.percentile(mse, 95)
        
        # Make binary predictions (anomaly vs normal)
        y_pred_anomaly = (mse > threshold).astype(int)
        y_true_binary = (y_test != 0).astype(int)  # 0 for normal, 1 for any attack
        
        detection_time = time.time() - start_time
        
        # Calculate metrics for binary classification
        accuracy = accuracy_score(y_true_binary, y_pred_anomaly)
        precision = precision_score(y_true_binary, y_pred_anomaly, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_anomaly, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_anomaly, zero_division=0)
        
        # Store results
        self.evaluator.results['Autoencoder'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': 0.0,  # Not applicable for this setup
            'Detection_Time': detection_time / len(X_test),
            'Predictions': y_pred_anomaly,
            'True_Labels': y_true_binary,
            'Threshold': threshold,
            'MSE': mse
        }
        
        # Store confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_anomaly)
        self.evaluator.confusion_matrices['Autoencoder'] = cm
        
        print(f"Autoencoder Results (Binary Anomaly Detection):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Detection Threshold: {threshold:.4f}")
        print(f"  Avg Detection Time: {detection_time/len(X_test)*1000:.2f} ms")
    
    def _plot_training_histories(self):
        """Plot training histories for deep learning models"""
        if not self.model_suite.training_history:
            return
        
        n_models = len(self.model_suite.training_history)
        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5*n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, history) in enumerate(self.model_suite.training_history.items()):
            # Plot training & validation accuracy
            axes[idx, 0].plot(history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history.history:
                axes[idx, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[idx, 0].set_title(f'{model_name} - Accuracy')
            axes[idx, 0].set_xlabel('Epoch')
            axes[idx, 0].set_ylabel('Accuracy')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True)
            
            # Plot training & validation loss
            axes[idx, 1].plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                axes[idx, 1].plot(history.history['val_loss'], label='Validation Loss')
            axes[idx, 1].set_title(f'{model_name} - Loss')
            axes[idx, 1].set_xlabel('Epoch')
            axes[idx, 1].set_ylabel('Loss')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_histories.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print(f"\n4.4 COMPREHENSIVE ANALYSIS REPORT")
        print("-" * 50)
        
        # Create detailed report
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE IDS ANALYSIS REPORT")
        report.append("AI-Enhanced Intrusion Detection Systems in Cloud Computing")
        report.append("="*80)
        
        # Dataset Summary
        report.append(f"\n4.4.1 Dataset Summary")
        report.append("-" * 30)
        data = self.processed_data[self.primary_dataset]
        report.append(f"Primary Dataset: {self.primary_dataset}")
        report.append(f"Training Samples: {len(data['X_train'])}")
        report.append(f"Testing Samples: {len(data['X_test'])}")
        report.append(f"Number of Features: {len(data['feature_names'])}")
        report.append(f"Number of Classes: {data['num_classes']}")
        report.append(f"Class Names: {', '.join(data['class_names'])}")
        
        # Model Performance Summary
        report.append(f"\n4.4.2 Model Performance Summary")
        report.append("-" * 35)
        
        if self.evaluator.results:
            # Create performance summary table
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Detection_Time']
            
            # Header
            header = f"{'Model':<20}"
            for metric in metrics:
                if metric == 'Detection_Time':
                    header += f"{'Det.Time(ms)':<12}"
                else:
                    header += f"{metric:<12}"
            report.append(header)
            report.append("-" * len(header))
            
            # Model results
            for model_name, results in self.evaluator.results.items():
                row = f"{model_name:<20}"
                for metric in metrics:
                    if metric == 'Detection_Time':
                        value = results[metric] * 1000  # Convert to ms
                        row += f"{value:<12.2f}"
                    else:
                        value = results[metric]
                        row += f"{value:<12.4f}"
                report.append(row)
        
        # Best Performing Models
        report.append(f"\n4.4.3 Best Performing Models")
        report.append("-" * 32)
        
        if self.evaluator.results:
            metrics_to_analyze = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for metric in metrics_to_analyze:
                best_model = max(self.evaluator.results.keys(), 
                               key=lambda x: self.evaluator.results[x][metric])
                best_score = self.evaluator.results[best_model][metric]
                report.append(f"Best {metric}: {best_model} ({best_score:.4f})")
            
            # Fastest model
            fastest_model = min(self.evaluator.results.keys(), 
                              key=lambda x: self.evaluator.results[x]['Detection_Time'])
            fastest_time = self.evaluator.results[fastest_model]['Detection_Time'] * 1000
            report.append(f"Fastest Detection: {fastest_model} ({fastest_time:.2f} ms)")
        
        # Key Findings and Insights
        report.append(f"\n4.4.4 Key Findings and Insights")
        report.append("-" * 33)
        
        insights = self._generate_insights()
        for insight in insights:
            report.append(f"• {insight}")
        
        # Recommendations
        report.append(f"\n4.4.5 Recommendations")
        report.append("-" * 23)
        
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"• {rec}")
        
        # Feature Importance (if available)
        if 'Random Forest' in self.trained_models:
            report.append(f"\n4.4.6 Feature Importance Analysis")
            report.append("-" * 35)
            
            rf_model = self.trained_models['Random Forest']
            feature_importance = rf_model.feature_importances_
            feature_names = data['feature_names']
            
            # Get top 10 features
            top_features_idx = np.argsort(feature_importance)[-10:][::-1]
            
            report.append("Top 10 Most Important Features (Random Forest):")
            for i, idx in enumerate(top_features_idx, 1):
                report.append(f"{i:2d}. {feature_names[idx]:<30} ({feature_importance[idx]:.4f})")
        
        # Conclusion
        report.append(f"\n4.4.7 Conclusion")
        report.append("-" * 17)
        report.append("This comprehensive analysis demonstrates the effectiveness of AI-enhanced")
        report.append("intrusion detection systems in cloud computing environments. The evaluation")
        report.append("of multiple machine learning and deep learning approaches provides valuable")
        report.append("insights into their respective strengths and limitations for cybersecurity")
        report.append("applications.")
        
        report.append(f"\nAnalysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        # Save report to file
        with open('IDS_Comprehensive_Analysis_Report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
        
        # Save results to pickle for future use
        results_data = {
            'processed_data': self.processed_data,
            'trained_models': {k: v for k, v in self.trained_models.items() 
                             if k not in ['LSTM', 'CNN', 'Hybrid CNN-LSTM', 'Autoencoder']},  # Exclude TF models
            'evaluation_results': self.evaluator.results,
            'confusion_matrices': self.evaluator.confusion_matrices,
            'training_histories': self.model_suite.training_history
        }
        
        with open('IDS_Analysis_Results.pkl', 'wb') as f:
            pickle.dump(results_data, f)
        
        print(f"\nResults saved to:")
        print(f"  - IDS_Comprehensive_Analysis_Report.txt")
        print(f"  - IDS_Analysis_Results.pkl")
        print(f"  - Various visualization files (.png)")
    
    def _generate_insights(self):
        """Generate key insights from the analysis"""
        insights = []
        
        if not self.evaluator.results:
            insights.append("No evaluation results available for insight generation.")
            return insights
        
        # Performance insights
        best_accuracy = max(self.evaluator.results.values(), key=lambda x: x['Accuracy'])['Accuracy']
        if best_accuracy > 0.95:
            insights.append("Achieved excellent detection accuracy (>95%) with advanced AI models.")
        elif best_accuracy > 0.90:
            insights.append("Demonstrated strong detection performance (>90%) across multiple models.")
        
        # Speed insights
        detection_times = [r['Detection_Time'] * 1000 for r in self.evaluator.results.values()]
        avg_detection_time = np.mean(detection_times)
        if avg_detection_time < 1.0:
            insights.append("Real-time detection capability achieved with sub-millisecond response times.")
        elif avg_detection_time < 10.0:
            insights.append("Near real-time detection performance suitable for production deployment.")
        
        # Model comparison insights
        traditional_models = ['Random Forest', 'SVM']
        deep_models = ['LSTM', 'CNN', 'Hybrid CNN-LSTM']
        
        trad_performance = [self.evaluator.results[m]['F1-Score'] for m in traditional_models 
                          if m in self.evaluator.results]
        deep_performance = [self.evaluator.results[m]['F1-Score'] for m in deep_models 
                          if m in self.evaluator.results]
        
        if trad_performance and deep_performance:
            if np.mean(deep_performance) > np.mean(trad_performance):
                insights.append("Deep learning models outperformed traditional ML approaches.")
            else:
                insights.append("Traditional ML models showed competitive performance with faster training.")
        
        # Class imbalance insights
        data = self.processed_data[self.primary_dataset]
        class_distribution = Counter(data['y_train'])
        if len(class_distribution) > 2:
            max_class_size = max(class_distribution.values())
            min_class_size = min(class_distribution.values())
            if max_class_size / min_class_size > 10:
                insights.append("Successfully handled severe class imbalance using SMOTE oversampling.")
        
        return insights
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        if not self.evaluator.results:
            recommendations.append("Complete model evaluation to generate specific recommendations.")
            return recommendations
        
        # Performance-based recommendations
        best_model = max(self.evaluator.results.keys(), 
                        key=lambda x: self.evaluator.results[x]['F1-Score'])
        recommendations.append(f"Deploy {best_model} for optimal overall performance in production.")
        
        # Speed-based recommendations
        fastest_model = min(self.evaluator.results.keys(), 
                          key=lambda x: self.evaluator.results[x]['Detection_Time'])
        if fastest_model != best_model:
            recommendations.append(f"Consider {fastest_model} for high-throughput environments requiring speed.")
        
        # Ensemble recommendations
        if len(self.evaluator.results) > 2:
            recommendations.append("Implement ensemble methods combining top-performing models for enhanced robustness.")
        
        # Deployment recommendations
        recommendations.append("Implement continuous learning to adapt to evolving attack patterns.")
        recommendations.append("Deploy distributed detection across multiple cloud nodes for scalability.")
        recommendations.append("Establish real-time monitoring and alert systems for immediate threat response.")
        
        # Data recommendations
        recommendations.append("Regularly update training data with latest attack signatures and patterns.")
        recommendations.append("Implement data preprocessing pipelines for consistent feature engineering.")
        
        return recommendations

# Main execution function
def main():
    """Main function to run the comprehensive IDS analysis"""
    print("Initializing Comprehensive IDS Analysis...")
    
    # Create analysis instance
    ids_analysis = ComprehensiveIDSAnalysis()
    
    # Run complete analysis
    ids_analysis.run_comprehensive_analysis()
    
    return ids_analysis

# Execute the analysis
if __name__ == "__main__":
    analysis_results = main()
        