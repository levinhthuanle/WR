"""
Demo script for the new Clothing Size Prediction Pipeline
Flow: Image + Height → YOLO Pose → Body Measurements → Random Forest Size Prediction
"""

from main import ClothingSizePredictionPipeline
import os

def demo_pipeline():
    """Demo the complete pipeline"""
    print("🚀 CLOTHING SIZE PREDICTION PIPELINE DEMO")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "models/yolo11n-pose.pt",
        "models/random_forest_model.pkl",
        "dataset/.jpg"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\nPlease make sure to:")
        print("1. Run 'python export_rf_weights.py' to create the Random Forest model")
        print("2. Download YOLO11n-pose model to models/ directory")
        print("3. Have test image in tests/ directory")
        return
    
    try:
        # Initialize pipeline
        pipeline = ClothingSizePredictionPipeline()
        
        # Demo with sample data
        test_cases = [
            {
                "image": "tests/thanh.jpg",
                "height": 170,
                "description": "Person 1 (170 cm)"
            },
            {
                "image": "tests/thanh.jpg", 
                "height": 160,
                "description": "Same person, different height (160 cm)"
            },
            {
                "image": "tests/thanh.jpg",
                "height": 180,
                "description": "Same person, different height (180 cm)"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*50}")
            print(f"TEST CASE {i+1}: {test_case['description']}")
            print("="*50)
            
            if not os.path.exists(test_case["image"]):
                print(f"❌ Test image not found: {test_case['image']}")
                continue
            
            # Run pipeline
            results = pipeline.run_complete_pipeline(
                image_path=test_case["image"],
                real_height_cm=test_case["height"],
                output_path=f"output/demo_result_{i+1}.jpg"
            )
            
            # Display results
            print(f"\n📊 RESULTS:")
            print(f"   🎯 Predicted Size: {results['predicted_size']}")
            
            print(f"\n   📐 Body Measurements:")
            for feature, value in results['measurements'].items():
                print(f"      • {feature}: {value:.1f} cm")
            
            print(f"\n   📈 Size Probabilities:")
            probabilities = results['size_probabilities']
            top_3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            for size, prob in top_3:
                print(f"      • {size}: {prob*100:.1f}%")
        
        print(f"\n✅ DEMO COMPLETED!")
        print(f"📁 Check output/ directory for annotated images")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def test_individual_steps():
    """Test each step of the pipeline individually"""
    print(f"\n🔍 TESTING INDIVIDUAL PIPELINE STEPS")
    print("=" * 50)
    
    try:
        pipeline = ClothingSizePredictionPipeline()
        image_path = "tests/thanh.jpg"
        height = 170
        
        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            return
        
        # Step 1: Extract keypoints
        print(f"\n1️⃣ TESTING KEYPOINT EXTRACTION")
        print("-" * 30)
        keypoints, _ = pipeline.extract_keypoints_from_image(image_path)
        print(f"Extracted keypoints:")
        for name, (x, y) in list(keypoints.items())[:5]:  # Show first 5
            print(f"   • {name}: ({x:.1f}, {y:.1f})")
        print(f"   ... and {len(keypoints)-5} more keypoints")
        
        # Step 2: Estimate measurements
        print(f"\n2️⃣ TESTING BODY MEASUREMENT ESTIMATION")
        print("-" * 30)
        measurements = pipeline.estimate_body_measurements(keypoints, height)
        
        # Step 3: Predict size
        print(f"\n3️⃣ TESTING SIZE PREDICTION")
        print("-" * 30)
        prediction, probabilities = pipeline.predict_clothing_size(measurements)
        print(f"Size prediction: {prediction}")
        
        print(f"\n✅ ALL STEPS WORKING CORRECTLY!")
        
    except Exception as e:
        print(f"❌ Individual step testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run demo
    demo_pipeline()
    
    # Ask if user wants to test individual steps
    print(f"\n" + "=" * 50)
    choice = input("Do you want to test individual pipeline steps? (y/n): ").lower().strip()
    if choice in ['y', 'yes']:
        test_individual_steps()
    
    print(f"\n✅ Demo completed!")