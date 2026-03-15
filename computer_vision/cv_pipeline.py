
import cv2
import numpy as np
from skimage.feature import hog
import os
import time

def preprocess_image(image):
    # Resize
    resized = cv2.resize(image,(128,128))
    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Noise Removal (Gaussian Blur)
    denoised = cv2.GaussianBlur(gray,(3,3),0)
    # Image Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    # Edge Detection
    edges = cv2.Canny(enhanced,50,150)
    return resized, gray, denoised, enhanced, edges

def extract_features(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(4,4),
        cells_per_block=(2,2),
        visualize=False
    )
    return features

def process_dataset(DATASET_PATH, PROCESSED_PATH, MAX_SAMPLE_IMAGES=None):
    data = []
    labels = []
    categories = ["debris","non_debris"]

    start_time = time.time()
    images_processed_count = 0

    for category in categories:
        path = os.path.join(DATASET_PATH, category)
        save_folder = os.path.join(PROCESSED_PATH, category)
        label = categories.index(category)

        print(f"\nProcessing: {category}")

        for img_name in os.listdir(path):
            if MAX_SAMPLE_IMAGES is not None and images_processed_count >= MAX_SAMPLE_IMAGES:
                break

            img_path = os.path.join(path, img_name)

            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load image {img_path}. Skipping.")
                    continue

                resized, gray, denoised, enhanced, edges = preprocess_image(image)

                os.makedirs(save_folder, exist_ok=True)
                save_path = os.path.join(save_folder, img_name)
                cv2.imwrite(save_path, edges)

                features = extract_features(edges)

                data.append(features)
                labels.append(label)
                images_processed_count += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                pass
        if MAX_SAMPLE_IMAGES is not None and images_processed_count >= MAX_SAMPLE_IMAGES:
            break

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"\nProcessed {images_processed_count} sample images in {time_taken:.2f} seconds.")

    if images_processed_count > 0:
        time_per_image = time_taken / images_processed_count
        # Assuming total_images_to_process for full dataset estimation
        total_images_to_process = 20000 + 40
        estimated_total_time = time_per_image * total_images_to_process

        print(f"Estimated time per image: {time_per_image:.4f} seconds")
        print(f"Estimated total time for {total_images_to_process} images: {estimated_total_time:.2f} seconds ({estimated_total_time/60:.2f} minutes, {estimated_total_time/3600:.2f} hours)")
    else:
        print("Could not process any images for estimation.")

    print(f"\nTotal Images Processed (for sample): {len(data)}")

    return np.array(data), np.array(labels)
