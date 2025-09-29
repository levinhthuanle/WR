from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import math

# Define the COCO keypoint labels (17 keypoints)
COCO_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

class BodyMeasurementPredictor:
    def __init__(self, pose_model_path="models/yolo11n-pose.pt", size_model_path=None):
        """
        Initialize the Body Measurement Predictor
        
        Args:
            pose_model_path: Path to YOLO pose detection model
            size_model_path: Path to trained size prediction model (optional)
        """
        self.pose_model = YOLO(pose_model_path)
        self.size_model = None
        self.scaler = StandardScaler()
        
        # Load or train size prediction model
        if size_model_path and os.path.exists(size_model_path):
            self.size_model = joblib.load(size_model_path)
        else:
            self._train_size_model()
    
    def _train_size_model(self):
        """Train the size prediction model using the dataset"""
        print("Training size prediction model...")
        
        # Load dataset
        df = pd.read_csv('clothing_size_dataset_synthetic_2000.csv')
        
        # Prepare features and target
        feature_columns = ['Shoulder Width', 'Belly', 'Neck Circumference', 'Hip Circumference', 'Shirt Length']
        X = df[feature_columns]
        y = df['Size']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest model (best performing from analysis)
        self.size_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.size_model.fit(X_scaled, y)
        
        # Save model
        joblib.dump(self.size_model, 'models/size_prediction_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        print("Model trained and saved successfully!")
    
    def extract_keypoints(self, image_path):
        """
        Extract keypoints from image using YOLO pose detection
        
        Args:
            image_path: Path to the input image
            
        Returns:
            keypoints: Dictionary of keypoint coordinates
            image: Processed image with annotations
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Run pose estimation
        results = self.pose_model(image)
        
        keypoints_dict = {}
        annotated_image = None
        
        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                # Get the first detected person
                person_keypoints = result.keypoints.xy[0].cpu().numpy()
                
                # Convert to dictionary
                for i, (x, y) in enumerate(person_keypoints):
                    if x > 0 and y > 0:  # Valid keypoint
                        keypoints_dict[COCO_KEYPOINTS[i]] = (float(x), float(y))
                
                # Get annotated image
                annotated_image = result.plot()
                break
        
        return keypoints_dict, annotated_image
    
    def calculate_pixel_height(self, keypoints):
        """
        Calculate height in pixels from keypoints
        
        Args:
            keypoints: Dictionary of keypoint coordinates
            
        Returns:
            height_pixels: Height in pixels
        """
        # Use top of head (approximate from nose/eyes) to ankle
        if "Nose" in keypoints and "Left Ankle" in keypoints:
            nose_y = keypoints["Nose"][1]
            ankle_y = keypoints["Left Ankle"][1]
            height_pixels = abs(ankle_y - nose_y)
        elif "Nose" in keypoints and "Right Ankle" in keypoints:
            nose_y = keypoints["Nose"][1]
            ankle_y = keypoints["Right Ankle"][1]
            height_pixels = abs(ankle_y - nose_y)
        else:
            raise ValueError("Cannot calculate height - missing required keypoints")
        
        return height_pixels
    
    def calculate_body_measurements(self, keypoints, real_height_cm):
        """
        Calculate body measurements from keypoints and real height
        
        Args:
            keypoints: Dictionary of keypoint coordinates
            real_height_cm: Real height of person in centimeters
            
        Returns:
            measurements: Dictionary of body measurements
        """
        # Calculate pixel to cm ratio
        height_pixels = self.calculate_pixel_height(keypoints)
        pixel_to_cm_ratio = real_height_cm / height_pixels
        
        measurements = {}
        
        # 1. Shoulder Width
        if "Left Shoulder" in keypoints and "Right Shoulder" in keypoints:
            left_shoulder = keypoints["Left Shoulder"]
            right_shoulder = keypoints["Right Shoulder"]
            shoulder_width_pixels = math.sqrt(
                (left_shoulder[0] - right_shoulder[0])**2 + 
                (left_shoulder[1] - right_shoulder[1])**2
            )
            measurements["Shoulder Width"] = shoulder_width_pixels * pixel_to_cm_ratio
        
        # 2. Hip Circumference (estimated from hip width)
        if "Left Hip" in keypoints and "Right Hip" in keypoints:
            left_hip = keypoints["Left Hip"]
            right_hip = keypoints["Right Hip"]
            hip_width_pixels = math.sqrt(
                (left_hip[0] - right_hip[0])**2 + 
                (left_hip[1] - right_hip[1])**2
            )
            # Convert width to circumference (approximate: circumference ≈ π * width)
            measurements["Hip Circumference"] = hip_width_pixels * pixel_to_cm_ratio * 2.5
        
        # 3. Neck Circumference (estimated from head size)
        if "Left Ear" in keypoints and "Right Ear" in keypoints:
            left_ear = keypoints["Left Ear"]
            right_ear = keypoints["Right Ear"]
            head_width_pixels = math.sqrt(
                (left_ear[0] - right_ear[0])**2 + 
                (left_ear[1] - right_ear[1])**2
            )
            # Neck circumference is approximately 0.6 times head width * π
            measurements["Neck Circumference"] = head_width_pixels * pixel_to_cm_ratio * 0.6 * 3.14159
        
        # 4. Belly (estimated from torso width at waist level)
        if ("Left Shoulder" in keypoints and "Right Shoulder" in keypoints and 
            "Left Hip" in keypoints and "Right Hip" in keypoints):
            # Calculate waist width as average of shoulder and hip widths
            shoulder_width_pixels = math.sqrt(
                (keypoints["Left Shoulder"][0] - keypoints["Right Shoulder"][0])**2 + 
                (keypoints["Left Shoulder"][1] - keypoints["Right Shoulder"][1])**2
            )
            hip_width_pixels = math.sqrt(
                (keypoints["Left Hip"][0] - keypoints["Right Hip"][0])**2 + 
                (keypoints["Left Hip"][1] - keypoints["Right Hip"][1])**2
            )
            waist_width_pixels = (shoulder_width_pixels + hip_width_pixels) / 2
            # Convert to circumference
            measurements["Belly"] = waist_width_pixels * pixel_to_cm_ratio * 2.8
        
        # 5. Shirt Length (shoulder to hip)
        if ("Left Shoulder" in keypoints and "Left Hip" in keypoints):
            shoulder = keypoints["Left Shoulder"]
            hip = keypoints["Left Hip"]
            shirt_length_pixels = math.sqrt(
                (shoulder[0] - hip[0])**2 + 
                (shoulder[1] - hip[1])**2
            )
            measurements["Shirt Length"] = shirt_length_pixels * pixel_to_cm_ratio
        elif ("Right Shoulder" in keypoints and "Right Hip" in keypoints):
            shoulder = keypoints["Right Shoulder"]
            hip = keypoints["Right Hip"]
            shirt_length_pixels = math.sqrt(
                (shoulder[0] - hip[0])**2 + 
                (shoulder[1] - hip[1])**2
            )
            measurements["Shirt Length"] = shirt_length_pixels * pixel_to_cm_ratio
        
        return measurements
    
    def predict_size(self, measurements):
        """
        Predict clothing size from body measurements
        
        Args:
            measurements: Dictionary of body measurements
            
        Returns:
            prediction: Predicted size
            probabilities: Prediction probabilities for each size
        """
        if self.size_model is None:
            raise ValueError("Size prediction model not loaded")
        
        # Prepare input features
        feature_order = ['Shoulder Width', 'Belly', 'Neck Circumference', 
                        'Hip Circumference', 'Shirt Length']
        
        # Check if all required measurements are available
        missing_features = [f for f in feature_order if f not in measurements]
        if missing_features:
            raise ValueError(f"Missing measurements: {missing_features}")
        
        input_data = np.array([[measurements[feature] for feature in feature_order]])
        
        # Scale the input
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        prediction = self.size_model.predict(input_scaled)[0]
        probabilities = self.size_model.predict_proba(input_scaled)[0]
        
        # Create probability dictionary
        prob_dict = {size: prob for size, prob in 
                    zip(self.size_model.classes_, probabilities)}
        
        return prediction, prob_dict
    
    def process_image_and_predict(self, image_path, real_height_cm, output_path=None):
        """
        Complete pipeline: extract keypoints, calculate measurements, predict size
        
        Args:
            image_path: Path to input image
            real_height_cm: Real height of person in centimeters
            output_path: Path to save annotated image (optional)
            
        Returns:
            results: Dictionary containing all results
        """
        print(f"Processing image: {image_path}")
        print(f"Real height: {real_height_cm} cm")
        
        # Extract keypoints
        keypoints, annotated_image = self.extract_keypoints(image_path)
        
        if not keypoints:
            raise ValueError("No person detected in the image")
        
        print(f"Detected {len(keypoints)} keypoints")
        
        # Calculate measurements
        measurements = self.calculate_body_measurements(keypoints, real_height_cm)
        
        print("\nCalculated measurements:")
        for measurement, value in measurements.items():
            print(f"{measurement}: {value:.2f} cm")
        
        # Predict size
        prediction, probabilities = self.predict_size(measurements)
        
        print(f"\nPredicted size: {prediction}")
        print("Size probabilities:")
        for size, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {size}: {prob:.4f} ({prob*100:.1f}%)")
        
        # Save annotated image if requested
        if output_path and annotated_image is not None:
            cv2.imwrite(output_path, annotated_image)
            print(f"\nAnnotated image saved to: {output_path}")
        
        # Compile results
        results = {
            "keypoints": keypoints,
            "measurements": measurements,
            "predicted_size": prediction,
            "size_probabilities": probabilities,
            "annotated_image": annotated_image
        }
        
        return results

def main():
    """Example usage"""
    # Initialize predictor
    predictor = BodyMeasurementPredictor()
    
    # Example: process an image
    try:
        # You need to provide the real height of the person in the image
        real_height = float(input("Enter the real height of the person (in cm): "))
        image_path = input("Enter image path (or press Enter for default 'tests/thanh.jpg'): ").strip()
        
        if not image_path:
            image_path = "tests/thanh.jpg"
        
        # Process image and get predictions
        results = predictor.process_image_and_predict(
            image_path=image_path,
            real_height_cm=real_height,
            output_path="output/body_measurement_result.jpg"
        )
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Predicted Size: {results['predicted_size']}")
        print("\nBody Measurements:")
        for measurement, value in results['measurements'].items():
            print(f"  {measurement}: {value:.2f} cm")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()