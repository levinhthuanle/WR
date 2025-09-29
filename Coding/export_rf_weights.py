"""
Simple script to train Random Forest and export weights
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_and_export_random_forest():
    """Train Random Forest model and export weights"""
    print("🚀 TRAINING RANDOM FOREST MODEL")
    print("=" * 50)
    
    # Load dataset
    print("📂 Loading dataset...")
    df = pd.read_csv('clothing_size_dataset_synthetic_2000.csv')
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Prepare features and target
    feature_columns = ['Shoulder Width', 'Belly', 'Neck Circumference', 'Hip Circumference', 'Shirt Length']
    X = df[feature_columns]
    y = df['Size']
    
    print(f"Features: {feature_columns}")
    print(f"Target classes: {sorted(y.unique())}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Random Forest
    print("\n🌳 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    train_accuracy = rf_model.score(X_train, y_train)
    test_predictions = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"✅ Training completed!")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/random_forest_model.pkl'
    joblib.dump(rf_model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Export weights and parameters
    export_weights(rf_model, X_train.columns, test_accuracy)
    
    # Test prediction
    test_prediction_example(rf_model, X_train.columns)
    
    return rf_model

def export_weights(model, feature_names, accuracy):
    """Export Random Forest weights to text file"""
    
    export_path = 'models/random_forest_weights.txt'
    
    with open(export_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RANDOM FOREST MODEL WEIGHTS AND PARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        
        # Model metadata
        f.write("📋 MODEL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Type: Random Forest Classifier\n")
        f.write(f"Number of Trees: {model.n_estimators}\n")
        f.write(f"Max Depth: {model.max_depth}\n")
        f.write(f"Random State: {model.random_state}\n")
        f.write(f"Number of Features: {model.n_features_in_}\n")
        f.write(f"Classes: {list(model.classes_)}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        
        # Feature importance (main weights)
        f.write("⭐ FEATURE IMPORTANCE (WEIGHTS):\n")
        f.write("-" * 40 + "\n")
        feature_importance = list(zip(feature_names, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance):
            f.write(f"{i+1:2d}. {feature:25s}: {importance:.6f}\n")
        
        f.write(f"\nMost Important Feature: {feature_importance[0][0]} ({feature_importance[0][1]:.4f})\n")
        f.write(f"Least Important Feature: {feature_importance[-1][0]} ({feature_importance[-1][1]:.4f})\n\n")
        
        # Tree statistics
        f.write("🌳 INDIVIDUAL TREE STATISTICS:\n")
        f.write("-" * 40 + "\n")
        
        tree_depths = []
        tree_nodes = []
        tree_leaves = []
        
        for i, tree in enumerate(model.estimators_):
            depth = tree.tree_.max_depth
            nodes = tree.tree_.node_count
            leaves = np.sum(tree.tree_.children_left == -1)
            
            tree_depths.append(depth)
            tree_nodes.append(nodes)
            tree_leaves.append(leaves)
            
            if i < 5:  # Show first 5 trees
                f.write(f"Tree {i+1:3d}: Depth={depth:2d}, Nodes={nodes:3d}, Leaves={leaves:3d}\n")
        
        f.write(f"...\n")
        f.write(f"Average Tree Depth: {np.mean(tree_depths):.2f}\n")
        f.write(f"Average Nodes per Tree: {np.mean(tree_nodes):.2f}\n")
        f.write(f"Average Leaves per Tree: {np.mean(tree_leaves):.2f}\n\n")
        
        # Decision rules summary (for first tree)
        f.write("🔍 SAMPLE DECISION RULES (Tree 1):\n")
        f.write("-" * 40 + "\n")
        
        tree = model.estimators_[0]
        export_tree_rules(tree, feature_names, model.classes_, f, max_rules=10)
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF EXPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"📊 Weights exported to: {export_path}")
    
    # Print feature importance to console
    print(f"\n⭐ FEATURE IMPORTANCE RANKING:")
    for i, (feature, importance) in enumerate(feature_importance):
        print(f"  {i+1}. {feature}: {importance:.4f}")

def export_tree_rules(tree, feature_names, classes, file_handle, max_rules=10):
    """Export sample decision rules from a tree"""
    
    tree_ = tree.tree_
    
    def recurse(node, depth, rule_path, rule_count):
        if rule_count >= max_rules:
            return rule_count
        
        if tree_.children_left[node] != tree_.children_right[node]:
            # Internal node
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            # Left child (<=)
            new_path = rule_path + f"{feature} <= {threshold:.2f}"
            rule_count = recurse(tree_.children_left[node], depth + 1, new_path, rule_count)
            
            # Right child (>)
            if rule_count < max_rules:
                new_path = rule_path + f"{feature} > {threshold:.2f}"
                rule_count = recurse(tree_.children_right[node], depth + 1, new_path, rule_count)
        else:
            # Leaf node
            if rule_count < max_rules:
                values = tree_.value[node][0]
                predicted_class_idx = np.argmax(values)
                predicted_class = classes[predicted_class_idx]
                samples = tree_.n_node_samples[node]
                
                file_handle.write(f"Rule {rule_count + 1}: IF {rule_path} THEN Size = {predicted_class} (samples: {samples})\n")
                rule_count += 1
        
        return rule_count
    
    recurse(0, 0, "", 0)

def test_prediction_example(model, feature_names):
    """Test the model with example data"""
    print(f"\n🧮 TESTING MODEL WITH EXAMPLE DATA:")
    print("-" * 40)
    
    # Example measurements
    examples = [
        {
            'Shoulder Width': 35.0,
            'Belly': 70.0,
            'Neck Circumference': 32.0,
            'Hip Circumference': 80.0,
            'Shirt Length': 65.0,
            'expected': 'S'
        },
        {
            'Shoulder Width': 42.0,
            'Belly': 85.0,
            'Neck Circumference': 37.0,
            'Hip Circumference': 92.0,
            'Shirt Length': 71.0,
            'expected': 'L'
        },
        {
            'Shoulder Width': 48.0,
            'Belly': 98.0,
            'Neck Circumference': 40.0,
            'Hip Circumference': 105.0,
            'Shirt Length': 76.0,
            'expected': 'XL'
        }
    ]
    
    for i, example in enumerate(examples):
        # Prepare input
        input_data = np.array([[example[feature] for feature in feature_names]])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        print(f"\nExample {i+1}:")
        print(f"  Input: {dict((k, v) for k, v in example.items() if k != 'expected')}")
        print(f"  Expected: {example['expected']}")
        print(f"  Predicted: {prediction}")
        print(f"  Confidence: {max(probabilities):.3f}")
        
        # Show top 2 probabilities
        prob_pairs = list(zip(model.classes_, probabilities))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top predictions:")
        for size, prob in prob_pairs[:2]:
            print(f"    {size}: {prob:.3f}")

def main():
    """Main function"""
    try:
        # Train and export model
        model = train_and_export_random_forest()
        
        print(f"\n✅ SUCCESS!")
        print(f"📁 Files created:")
        print(f"  • models/random_forest_model.pkl (trained model)")
        print(f"  • models/random_forest_weights.txt (weights export)")
        
    except FileNotFoundError:
        print("❌ Error: clothing_size_dataset_synthetic_2000.csv not found!")
        print("Make sure the dataset file is in the current directory.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()