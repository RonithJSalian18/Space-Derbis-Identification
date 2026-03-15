# ==========================================
# SPACE DEBRIS DETECTION PROJECT
# FULL DATASET CV PIPELINE
# ==========================================


# ==============================
# 1. Mount Google Drive
# ==============================

# from google.colab import drive
# drive.mount('/content/drive')


# ==============================
# 2. Extract Dataset ZIP
# ==============================

import zipfile
import os

dataset_zip = "/content/drive/MyDrive/space_debris_project/dataset.zip"
extract_path = "/content/dataset"

with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset Extracted Successfully")


# ==============================
# 3. Import Libraries
# ==============================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog


# ==============================
# 4. Dataset Paths
# ==============================

DATASET_PATH = "/content/dataset"

PROCESSED_PATH = "/content/processed_dataset"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH + "/debris", exist_ok=True)
os.makedirs(PROCESSED_PATH + "/non_debris", exist_ok=True)


# ==============================
# 5. CV Preprocessing Function
# ==============================

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


# ==============================
# 6. Feature Extraction (HOG)
# ==============================

def extract_features(image):

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(4,4),
        cells_per_block=(2,2),
        visualize=False
    )

    return features


# ==============================
# 7. Test CV Pipeline on One Image
# ==============================

sample_path = "/content/dataset/debris/" + os.listdir("/content/dataset/debris")[0]

image = cv2.imread(sample_path)

resized, gray, denoised, enhanced, edges = preprocess_image(image)

plt.figure(figsize=(15,5))

plt.subplot(1,5,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,5,2)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,5,3)
plt.title("Noise Removed")
plt.imshow(denoised, cmap="gray")
plt.axis("off")

plt.subplot(1,5,4)
plt.title("Enhanced")
plt.imshow(enhanced, cmap="gray")
plt.axis("off")

plt.subplot(1,5,5)
plt.title("Edges")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.show()


# ==============================
# 8. Process Entire Dataset
# ==============================

data = []
labels = []

categories = ["debris","non_debris"]

for category in categories:

    path = os.path.join(DATASET_PATH,category)

    save_folder = os.path.join(PROCESSED_PATH,category)

    label = categories.index(category)

    print("\nProcessing:", category)

    for img in os.listdir(path):

        img_path = os.path.join(path,img)

        try:

            image = cv2.imread(img_path)

            resized, gray, denoised, enhanced, edges = preprocess_image(image)

            # Save processed edge image
            save_path = os.path.join(save_folder,img)
            cv2.imwrite(save_path,edges)

            # Extract HOG features
            features = extract_features(edges)

            data.append(features)
            labels.append(label)

        except:
            pass


print("\nTotal Images Processed:",len(data))


# ==============================
# 9. Convert Dataset to NumPy
# ==============================

X = np.array(data)
y = np.array(labels)

print("\nDataset Converted to NumPy")

print("Feature Matrix Shape:",X.shape)
print("Label Vector Shape:",y.shape)


# ==============================
# 10. Show Sample Feature Values
# ==============================

active_indices = np.where(X[0] > 0)[0]

if len(active_indices) > 0:

    start = active_indices[0]

    print("\nSample Feature Values:")
    print(X[0][start:start+20])

else:

    print("\nWarning: All features are zero. Check preprocessing.")