import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class LoanApprovalModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the loan dataset"""
        df = pd.read_csv(csv_path)
        
        # Clean column names and data
        df.columns = df.columns.str.strip()
        
        # Clean string columns by removing leading/trailing whitespace
        string_cols = ['education', 'self_employed', 'loan_status']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].str.strip()
        
        # Encode categorical variables
        categorical_cols = ['education', 'self_employed']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Encode target variable
        target_le = LabelEncoder()
        df['loan_status'] = target_le.fit_transform(df['loan_status'])
        self.label_encoders['loan_status'] = target_le
        
        return df
    
    def train_test_split_data(self, df, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X = df.drop(['loan_id', 'loan_status'], axis=1)
        y = df['loan_status']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """Train the model"""
        # Scale numerical features
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        print("Model trained successfully!")
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                  target_names=self.label_encoders['loan_status'].classes_))
        
        return accuracy, predictions
    
    def predict_single(self, loan_data):
        """Predict for a single loan application"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Convert to DataFrame if it's a dict
        if isinstance(loan_data, dict):
            loan_data = pd.DataFrame([loan_data])
        
        # Create a copy to avoid modifying original
        loan_data_encoded = loan_data.copy()
        
        # Remove loan_status if present (target variable shouldn't be in features)
        if 'loan_status' in loan_data_encoded.columns:
            loan_data_encoded = loan_data_encoded.drop(['loan_status'], axis=1)
        
        # Remove loan_id if present
        if 'loan_id' in loan_data_encoded.columns:
            loan_data_encoded = loan_data_encoded.drop(['loan_id'], axis=1)
        
        # Encode categorical variables only if they are strings
        for col in ['education', 'self_employed']:
            if col in loan_data_encoded.columns:
                # Check if already encoded (numeric)
                if loan_data_encoded[col].dtype == 'object':
                    # Clean any whitespace
                    loan_data_encoded[col] = loan_data_encoded[col].str.strip()
                    loan_data_encoded[col] = self.label_encoders[col].transform(loan_data_encoded[col])
        
        # Ensure column order matches training data
        expected_columns = ['no_of_dependents', 'education', 'self_employed', 'income_annum', 
                           'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 
                           'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
        
        # Reorder columns to match training data
        loan_data_encoded = loan_data_encoded[expected_columns]
        
        # Scale the data
        loan_data_scaled = self.scaler.transform(loan_data_encoded)
        
        # Make prediction
        prediction = self.model.predict(loan_data_scaled)[0]
        probability = self.model.predict_proba(loan_data_scaled)[0]
        
        # Decode prediction
        prediction_label = self.label_encoders['loan_status'].inverse_transform([prediction])[0]
        
        return {
            'prediction': prediction_label,
            'confidence': max(probability),
            'probability_approved': probability[1] if len(probability) > 1 else probability[0]
        }
    
    def save_model(self, model_path):
        """Save the trained model and encoders"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model and encoders"""
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.scaler = model_data['scaler']
            self.is_trained = True
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

def main():
    # Initialize model
    model = LoanApprovalModel()
    
    # Use relative paths
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'loan_approval_dataset 2.csv')
    model_path = os.path.join(base_dir, 'loan_model.pkl')
    test_data_path = os.path.join(base_dir, 'test_data.csv')
    
    # Load and preprocess data
    df = model.load_and_preprocess_data(csv_path)
    
    # Split data
    X_train, X_test, y_train, y_test = model.train_test_split_data(df)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate model
    accuracy, predictions = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model(model_path)
    
    # Save test data for later use with LLM agents (exclude target variable for clean prediction)
    test_data_for_llm = X_test.copy()
    
    # Convert y_test back to string labels for clarity
    # y_test uses model encoding: 0=Approved, 1=Rejected
    # Convert to clear string labels
    y_test_labels = model.label_encoders['loan_status'].inverse_transform(y_test)
    test_data_for_llm['true_loan_status_string'] = y_test_labels
    
    # For compatibility with existing code, also keep encoded version with correct interpretation
    # Convert model encoding (0=Approved, 1=Rejected) to logical encoding (0=Rejected, 1=Approved)
    y_test_logical = 1 - y_test  # Flip the encoding
    test_data_for_llm['true_loan_status'] = y_test_logical
    
    test_data_for_llm.to_csv(test_data_path, index=False)
    
    print(f"\nTest data saved with {len(test_data_for_llm)} samples for LLM agent evaluation")
    print(f"Test data columns: {list(test_data_for_llm.columns)}")
    print(f"Label encoding: 0=Rejected, 1=Approved (logical encoding)")
    print(f"String labels also included for clarity")
    
    return model, test_data_for_llm

if __name__ == "__main__":
    model, test_data = main()