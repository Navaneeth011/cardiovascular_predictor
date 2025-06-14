import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class CardioPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                               'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        
    def load_and_preprocess_data(self, csv_path='cardio_train.csv'):
        """Load and preprocess the cardiovascular dataset"""
        try:
            df = pd.read_csv(csv_path, delimiter=';')
            print(f"Original dataset shape: {df.shape}")
            
            # Convert age from days to years
            df['age'] = df['age'] / 365.25
            
            # Data cleaning - remove unrealistic values
            # Remove impossible blood pressure values
            df = df[
                (df['ap_hi'] >= 60) & (df['ap_hi'] <= 250) &
                (df['ap_lo'] >= 40) & (df['ap_lo'] <= 150) &
                (df['ap_hi'] > df['ap_lo'])  # Systolic should be higher than diastolic
            ]
            
            # Remove unrealistic height and weight values
            df = df[
                (df['height'] >= 100) & (df['height'] <= 250) &
                (df['weight'] >= 30) & (df['weight'] <= 200)
            ]
            
            # Remove unrealistic age values
            df = df[(df['age'] >= 18) & (df['age'] <= 100)]
            
            print(f"Cleaned dataset shape: {df.shape}")
            
            # Calculate dataset statistics for insights
            self.dataset_stats = {
                'avg_weight': df['weight'].mean(),
                'avg_ap_hi': df['ap_hi'].mean(),
                'avg_ap_lo': df['ap_lo'].mean(),
                'avg_cholesterol': df['cholesterol'].mean(),
                'avg_gluc': df['gluc'].mean(),
                'cardio_rate': df['cardio'].mean()
            }
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def train_model(self, df):
        """Train Random Forest model with cross-validation"""
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['cardio']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        # Train Random Forest with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        self.feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 Most Important Features:")
        for feature, importance in self.feature_importance[:5]:
            print(f"{feature}: {importance:.4f}")
        
        return accuracy >= 0.70
    
    def save_model(self, model_path='cardio_model.pkl'):
        """Save the trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'dataset_stats': self.dataset_stats,
            'feature_importance': self.feature_importance
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='cardio_model.pkl'):
        """Load the trained model and scaler"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.dataset_stats = model_data['dataset_stats']
            self.feature_importance = model_data['feature_importance']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, user_input):
        """Make prediction for user input"""
        if self.model is None:
            return None
        
        # Prepare input data
        input_df = pd.DataFrame([user_input], columns=self.feature_columns)
        
        # Scale numerical features
        numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        input_scaled = input_df.copy()
        input_scaled[numerical_features] = self.scaler.transform(input_df[numerical_features])
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probability': float(probability[1]),  # Probability of cardio=1
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
        }

def main():
    """Train and save the model"""
    predictor = CardioPredictor()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data()
    if df is None:
        print("Failed to load data. Make sure cardio_train.csv is in the current directory.")
        return
    
    # Train model
    success = predictor.train_model(df)
    if success:
        predictor.save_model()
        print("Model training completed successfully!")
    else:
        print("Model accuracy is below 70%. Please check the data and model parameters.")

if __name__ == "__main__":
    main()