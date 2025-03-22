from ultralytics import YOLO
import cv2
import numpy as np

# Define the COCO keypoint labels (17 keypoints)
COCO_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def process_image(image_path, output_path="output_pose.jpg"):
    model = YOLO("yolo11n-pose.pt")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Run pose estimation
    results = model(image)

    for idx, result in enumerate(results):
        keypoints = result.keypoints.xy  # Shape: (N, 17, 2), N = number of persons

        # Print coordinates for each detected person
        for person_idx, person_keypoints in enumerate(keypoints):
            print(f"\nPerson {person_idx + 1}:")
            person_keypoints = person_keypoints.cpu().numpy()  # Convert to numpy
            for kp_idx, (x, y) in enumerate(person_keypoints):
                label = COCO_KEYPOINTS[kp_idx]
                if x > 0 and y > 0:  # Check if keypoint is detected (non-zero)
                    print(f"{label}: (x: {x:.2f}, y: {y:.2f})")
                else:
                    print(f"{label}: Not detected")

        annotated_image = result.plot()

        cv2.imwrite(output_path, annotated_image)
        print(f"\nOutput image saved as: {output_path}")

def main():
    input_image = "test.jpg"
    output_image = "output_pose.jpg"

    try:
        process_image(input_image, output_image)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()