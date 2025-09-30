from ultralytics import YOLO
import cv2
import numpy as np
import joblib
import os
import math
import sys

# Define the COCO keypoint labels (17 keypoints)
COCO_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

class ClothingSizePredictionPipeline:
    def __init__(self, pose_model_path="models/yolo11n-pose.pt", rf_model_path="models/random_forest_model.pkl"):
        """
        Initialize the complete prediction pipeline
        
        Args:
            pose_model_path: Path to YOLO pose detection model
            rf_model_path: Path to trained Random Forest model
        """
        print("🚀 Initializing Clothing Size Prediction Pipeline...")
        
        # Load YOLO pose detection model
        print("📥 Loading YOLO pose detection model...")
        self.pose_model = YOLO(pose_model_path)
        print("✅ YOLO model loaded successfully!")
        
        # Load Random Forest model
        print("📥 Loading Random Forest clothing size model...")
        if not os.path.exists(rf_model_path):
            raise FileNotFoundError(f"Random Forest model not found: {rf_model_path}")
        
        self.rf_model = joblib.load(rf_model_path)
        print("✅ Random Forest model loaded successfully!")
        
        # Feature names for Random Forest
        self.feature_names = ['Shoulder Width', 'Belly', 'Neck Circumference', 'Hip Circumference', 'Shirt Length']
    
    def extract_keypoints_from_image(self, image_path):
        """
        Step 1: Extract keypoints from image using YOLO pose detection
        
        Args:
            image_path: Path to input image
            
        Returns:
            keypoints_dict: Dictionary of keypoint coordinates
            annotated_image: Image with pose annotations
        """
        print("👁️  Step 1: Extracting keypoints from image...")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Run YOLO pose estimation
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
        
        if not keypoints_dict:
            raise ValueError("No person detected in the image!")
        
        print(f"✅ Detected {len(keypoints_dict)} keypoints")
        return keypoints_dict, annotated_image
    
    def estimate_body_measurements(self, keypoints, real_height_cm):
        """
        Step 2: Estimate body measurements from keypoints and real height
        
        Args:
            keypoints: Dictionary of keypoint coordinates
            real_height_cm: Real height of person in centimeters
            
        Returns:
            measurements: Dictionary of estimated body measurements
        """
        print("📏 Step 2: Estimating body measurements...")
        
        # Calculate pixel to cm ratio
        height_pixels = self.calculate_pixel_height(keypoints)
        pixel_to_cm_ratio = real_height_cm / height_pixels
        
        print(f"   • Height in pixels: {height_pixels:.1f}")
        print(f"   • Pixel to cm ratio: {pixel_to_cm_ratio:.4f}")
        
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
            print(f"   • Shoulder Width: {measurements['Shoulder Width']:.1f} cm")
        
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
            print(f"   • Hip Circumference: {measurements['Hip Circumference']:.1f} cm")
        
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
            print(f"   • Neck Circumference: {measurements['Neck Circumference']:.1f} cm")
        
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
            print(f"   • Belly Circumference: {measurements['Belly']:.1f} cm")
        
        # 5. Shirt Length (shoulder to hip)
        if ("Left Shoulder" in keypoints and "Left Hip" in keypoints):
            shoulder = keypoints["Left Shoulder"]
            hip = keypoints["Left Hip"]
            shirt_length_pixels = math.sqrt(
                (shoulder[0] - hip[0])**2 + 
                (shoulder[1] - hip[1])**2
            )
            measurements["Shirt Length"] = shirt_length_pixels * pixel_to_cm_ratio
            print(f"   • Shirt Length: {measurements['Shirt Length']:.1f} cm")
        elif ("Right Shoulder" in keypoints and "Right Hip" in keypoints):
            shoulder = keypoints["Right Shoulder"]
            hip = keypoints["Right Hip"]
            shirt_length_pixels = math.sqrt(
                (shoulder[0] - hip[0])**2 + 
                (shoulder[1] - hip[1])**2
            )
            measurements["Shirt Length"] = shirt_length_pixels * pixel_to_cm_ratio
            print(f"   • Shirt Length: {measurements['Shirt Length']:.1f} cm")
        
        # Check if all required measurements are available
        missing_measurements = [f for f in self.feature_names if f not in measurements]
        if missing_measurements:
            raise ValueError(f"Could not estimate required measurements: {missing_measurements}")
        
        print("✅ Body measurements estimated successfully!")
        return measurements
    
    def calculate_pixel_height(self, keypoints):
        """Calculate height in pixels from keypoints"""
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
            raise ValueError("Cannot calculate height - missing required keypoints (Nose and Ankle)")
        
        return height_pixels
    
    def predict_clothing_size(self, measurements):
        """
        Step 3: Predict clothing size using Random Forest model
        
        Args:
            measurements: Dictionary of body measurements
            
        Returns:
            prediction: Predicted clothing size
            probabilities: Probability for each size
        """
        print("🧠 Step 3: Predicting clothing size with Random Forest...")
        
        # Prepare input features in correct order
        input_data = np.array([[measurements[feature] for feature in self.feature_names]])
        
        # Make prediction
        prediction = self.rf_model.predict(input_data)[0]
        probabilities = self.rf_model.predict_proba(input_data)[0]
        
        # Create probability dictionary
        prob_dict = {size: prob for size, prob in 
                    zip(self.rf_model.classes_, probabilities)}
        
        print(f"✅ Predicted size: {prediction} (confidence: {max(probabilities):.3f})")
        
        return prediction, prob_dict
    
    def run_complete_pipeline(self, image_path, real_height_cm, output_path="output/clothing_size_result.jpg"):
        """
        Run the complete pipeline: YOLO → Body Measurements → Random Forest Prediction
        
        Args:
            image_path: Path to input image
            real_height_cm: Real height of person in centimeters
            output_path: Path to save annotated image
            
        Returns:
            results: Complete results dictionary
        """
        print("🎯 STARTING COMPLETE CLOTHING SIZE PREDICTION PIPELINE")
        print("=" * 60)
        
        # Step 1: Extract keypoints using YOLO
        keypoints, annotated_image = self.extract_keypoints_from_image(image_path)
        
        # Step 2: Estimate body measurements
        measurements = self.estimate_body_measurements(keypoints, real_height_cm)
        
        # Step 3: Predict clothing size using Random Forest
        predicted_size, size_probabilities = self.predict_clothing_size(measurements)
        
        # Save annotated image
        if output_path and annotated_image is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            print(f"💾 Annotated image saved to: {output_path}")
        
        # Compile results
        results = {
            "keypoints": keypoints,
            "measurements": measurements,
            "predicted_size": predicted_size,
            "size_probabilities": size_probabilities,
            "annotated_image_path": output_path
        }
        
        return results

def main():
    """
    Main function implementing the new pipeline:
    1. User provides image + real height
    2. YOLO extracts pose keypoints
    3. Calculate body measurements from keypoints + height
    4. Random Forest predicts clothing size from measurements
    """
    print("=" * 60)
    print("🎯 CLOTHING SIZE PREDICTION PIPELINE")
    print("=" * 60)
    print("Flow: Image + Height → YOLO Pose → Body Measurements → RF Size Prediction")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = ClothingSizePredictionPipeline()
        print("✅ Pipeline initialized successfully!\n")
        
        # Get user inputs
        if len(sys.argv) >= 3:
            # Command line arguments
            image_path = sys.argv[1]
            real_height = float(sys.argv[2])
        else:
            # Interactive input
            image_path = input("📁 Enter image path (or press Enter for 'tests/thanh.jpg'): ").strip()
            if not image_path:
                image_path = "dataset/pig_1.jpg"
            
            real_height = float(input("📏 Enter real height of person (cm): "))
        
        print(f"\n📋 INPUT:")
        print(f"   • Image: {image_path}")
        print(f"   • Real Height: {real_height} cm")
        print("-" * 40)
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            image_path=image_path,
            real_height_cm=real_height,
            output_path="output/clothing_size_prediction.jpg"
        )
        
        # Display results
        print("\n" + "="*60)
        print("🎉 FINAL PREDICTION RESULTS")
        print("="*60)
        
        print(f"📏 PREDICTED SIZE: {results['predicted_size']}")
        
        print("\n📐 ESTIMATED BODY MEASUREMENTS:")
        measurements = results['measurements']
        for feature in pipeline.feature_names:
            print(f"   • {feature}: {measurements[feature]:.1f} cm")
        
        print("\n📊 SIZE PREDICTION CONFIDENCE:")
        probabilities = results['size_probabilities']
        for size, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {size}: {prob*100:.1f}%")
        
        print(f"\n💾 Results saved to: {results['annotated_image_path']}")
        print("="*60)
        
        return results
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("Make sure the image file and model files exist.")
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()