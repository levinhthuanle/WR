import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClothingSizePredictor:
    def __init__(self, csv_file_path):
        """Initialize the predictor with dataset"""
        self.csv_file_path = csv_file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.predictions = {}
        self.model_scores = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("=" * 60)
        print("LOADING AND PREPARING DATA")
        print("=" * 60)
        
        # Load dataset
        self.df = pd.read_csv(self.csv_file_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nDataset info:")
        print(self.df.info())
        
        # Display basic statistics
        print(f"\nSize distribution:")
        print(self.df['Size'].value_counts().sort_index())
        
        # Prepare features and target
        feature_columns = ['Shoulder Width', 'Belly', 'Neck Circumference', 'Hip Circumference', 'Shirt Length']
        X = self.df[feature_columns]
        y = self.df['Size']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        # Scale features for models that need it
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Encode labels for XGBoost
        self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "=" * 60)
        print("TRAINING LOGISTIC REGRESSION")
        print("=" * 60)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train_scaled, self.y_train)
        
        predictions = model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, predictions)
        
        self.models['Logistic Regression'] = model
        self.predictions['Logistic Regression'] = predictions
        self.model_scores['Logistic Regression'] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions))
        
        return model, predictions
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "=" * 60)
        print("TRAINING RANDOM FOREST")
        print("=" * 60)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        
        self.models['Random Forest'] = model
        self.predictions['Random Forest'] = predictions
        self.model_scores['Random Forest'] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return model, predictions
    
    def train_svm(self):
        """Train SVM model"""
        print("\n" + "=" * 60)
        print("TRAINING SVM (RBF KERNEL)")
        print("=" * 60)
        
        model = SVC(kernel='rbf', random_state=42, gamma='scale')
        model.fit(self.X_train_scaled, self.y_train)
        
        predictions = model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, predictions)
        
        self.models['SVM'] = model
        self.predictions['SVM'] = predictions
        self.model_scores['SVM'] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions))
        
        return model, predictions
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "=" * 60)
        print("TRAINING XGBOOST")
        print("=" * 60)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        model.fit(self.X_train, self.y_train_encoded)
        
        predictions_encoded = model.predict(self.X_test)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        accuracy = accuracy_score(self.y_test, predictions)
        
        self.models['XGBoost'] = model
        self.predictions['XGBoost'] = predictions
        self.model_scores['XGBoost'] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions))
        
        return model, predictions
    
    def perform_cross_validation(self):
        """Perform cross-validation for all models"""
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 60)
        
        cv_results = {}
        
        # Logistic Regression CV
        lr_cv = cross_val_score(self.models['Logistic Regression'], 
                               self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
        cv_results['Logistic Regression'] = lr_cv
        
        # Random Forest CV
        rf_cv = cross_val_score(self.models['Random Forest'], 
                               self.X_train, self.y_train, cv=5, scoring='accuracy')
        cv_results['Random Forest'] = rf_cv
        
        # SVM CV
        svm_cv = cross_val_score(self.models['SVM'], 
                                self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
        cv_results['SVM'] = svm_cv
        
        # XGBoost CV
        xgb_cv = cross_val_score(self.models['XGBoost'], 
                                self.X_train, self.y_train_encoded, cv=5, scoring='accuracy')
        cv_results['XGBoost'] = xgb_cv
        
        # Print results
        for model_name, scores in cv_results.items():
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"{model_name:20}: {mean_score:.4f} (+/- {2*std_score:.4f})")
        
        return cv_results
    
    def compare_models(self):
        """Compare all models and visualize results"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        # Print accuracy comparison
        print("Test Set Accuracy Comparison:")
        print("-" * 40)
        for model_name, accuracy in sorted(self.model_scores.items(), 
                                         key=lambda x: x[1], reverse=True):
            print(f"{model_name:20}: {accuracy:.4f}")
        
        # Find best model
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        best_accuracy = self.model_scores[best_model_name]
        print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return best_model_name
    
    def plot_results(self):
        """Create visualization plots"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clothing Size Prediction - Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison bar plot
        ax1 = axes[0, 0]
        models = list(self.model_scores.keys())
        accuracies = list(self.model_scores.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (model, acc) in enumerate(zip(models, accuracies)):
            ax1.text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontweight='bold')
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Confusion matrix for best model
        ax2 = axes[0, 1]
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        best_predictions = self.predictions[best_model_name]
        
        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(self.df['Size'].unique()), 
                   yticklabels=sorted(self.df['Size'].unique()),
                   ax=ax2)
        ax2.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        # 3. Feature importance (for Random Forest)
        ax3 = axes[1, 0]
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_names = self.X_train.columns
            importances = rf_model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            ax3.bar(range(len(importances)), importances[indices], 
                   color='#FFA07A', alpha=0.8, edgecolor='black', linewidth=1)
            ax3.set_title('Feature Importance (Random Forest)', fontweight='bold')
            ax3.set_ylabel('Importance')
            ax3.set_xticks(range(len(importances)))
            ax3.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
        
        # 4. Size distribution in dataset
        ax4 = axes[1, 1]
        size_counts = self.df['Size'].value_counts().sort_index()
        colors_pie = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        
        wedges, texts, autotexts = ax4.pie(size_counts.values, labels=size_counts.index, 
                                          autopct='%1.1f%%', startangle=90, colors=colors_pie)
        ax4.set_title('Size Distribution in Dataset', fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Model comparison with error bars (cross-validation)
        self.plot_cv_comparison()
    
    def plot_cv_comparison(self):
        """Plot cross-validation comparison"""
        # Perform cross-validation
        cv_results = self.perform_cross_validation()
        
        plt.figure(figsize=(12, 8))  # Tăng chiều cao figure
        
        models = list(cv_results.keys())
        means = [scores.mean() for scores in cv_results.values()]
        stds = [scores.std() for scores in cv_results.values()]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        plt.bar(models, means, yerr=stds, capsize=5, color=colors, 
            alpha=0.8, edgecolor='black', linewidth=1)
        plt.title('Cross-Validation Accuracy Comparison (5-Fold)', 
                fontsize=14, fontweight='bold', pad=30)  # Tăng padding của title
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)  # Tăng ylim để có thêm không gian phía trên
        plt.grid(True, alpha=0.3)
        
        # Điều chỉnh vị trí labels xuống thấp hơn
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i, mean + std + 0.015, f'{mean:.4f}±{std:.3f}', 
                    ha='center', fontweight='bold', fontsize=10)  # Giảm font size và điều chỉnh vị trí
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(pad=3.0)  # Tăng padding cho tight_layout
        plt.subplots_adjust(top=0.9)  # Điều chỉnh margin phía trên
        plt.show()
    
    def predict_new_sample(self, measurements, model_name=None):
        """Predict size for new measurements"""
        if model_name is None:
            model_name = max(self.model_scores, key=self.model_scores.get)
        
        model = self.models[model_name]
        
        # Prepare input
        if isinstance(measurements, dict):
            feature_order = ['Shoulder Width', 'Belly', 'Neck Circumference', 
                           'Hip Circumference', 'Shirt Length']
            input_data = np.array([[measurements[feature] for feature in feature_order]])
        else:
            input_data = np.array([measurements])
        
        # Scale if needed
        if model_name in ['Logistic Regression', 'SVM']:
            input_data = self.scaler.transform(input_data)
        
        # Predict
        if model_name == 'XGBoost':
            prediction_encoded = model.predict(input_data)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)[0]
        else:
            prediction = model.predict(input_data)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            if model_name == 'XGBoost':
                probabilities = model.predict_proba(input_data)[0]
                prob_dict = {size: prob for size, prob in 
                           zip(self.label_encoder.classes_, probabilities)}
            else:
                if model_name in ['Logistic Regression', 'SVM']:
                    probabilities = model.predict_proba(input_data)[0]
                else:
                    probabilities = model.predict_proba(input_data)[0]
                prob_dict = {size: prob for size, prob in 
                           zip(model.classes_, probabilities)}
            
            return prediction, prob_dict
        
        return prediction, None
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 STARTING CLOTHING SIZE PREDICTION ANALYSIS 🚀")
        print("=" * 60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train all models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_svm()
        self.train_xgboost()
        
        # Compare models
        best_model = self.compare_models()
        
        # Perform cross-validation
        self.perform_cross_validation()
        
        # Create visualizations
        self.plot_results()
        
        print(f"\n🎉 ANALYSIS COMPLETE! Best model: {best_model} 🎉")
        
        return best_model

# Example usage and testing
def main():
    # Initialize predictor
    predictor = ClothingSizePredictor('clothing_size_dataset_synthetic_2000.csv')
    
    # Run complete analysis
    best_model = predictor.run_complete_analysis()
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    
    # Example measurements
    example_measurements = {
        'Shoulder Width': 40.5,
        'Belly': 85.2,
        'Neck Circumference': 37.1,
        'Hip Circumference': 90.3,
        'Shirt Length': 70.5
    }
    
    prediction, probabilities = predictor.predict_new_sample(example_measurements)
    
    print(f"Input measurements: {example_measurements}")
    print(f"Predicted size: {prediction}")
    
    if probabilities:
        print("\nPrediction probabilities:")
        for size, prob in sorted(probabilities.items()):
            print(f"  {size}: {prob:.4f}")

if __name__ == "__main__":
    main()