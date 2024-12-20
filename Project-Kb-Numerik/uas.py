try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2  # OpenCV untuk pemrosesan gambar
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError as e:
    print(f"Error importing modules: {e}")
    exit()

# Fungsi untuk memuat gambar menggunakan OpenCV dan mengubah ukurannya
def load_and_preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)  # Membaca gambar
    image = cv2.resize(image, target_size)  # Mengubah ukuran gambar
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi BGR ke RGB
    image = image / 255.0  # Normalisasi nilai piksel
    return image

# Fungsi untuk mengklasifikasikan gambar baru
def classify_image(image_path, model, target_size):
    image = load_and_preprocess_image(image_path, target_size)
    image = image.reshape(1, -1)  # Bentuk ulang gambar untuk prediksi
    prediction = model.predict(image)
    return categories[prediction[0]]



dataset_path = 'dataset/'  
categories = ['cats', 'dogs']
data = []
labels = []

# Muat dan preprocess gambar
for category in categories:
    category_path = os.path.join(dataset_path, category)
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        try:
            image = load_and_preprocess_image(image_path, target_size=(64, 64))
            data.append(image)
            labels.append(categories.index(category))
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")

# Ubah data dan label ke dalam array numpy
data = np.array(data)
labels = np.array(labels)

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Bentuk ulang data untuk KNN (reshape ke array 1D)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Definisikan dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi model
y_pred = knn.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualisasi hasil Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(categories))
plt.xticks(tick_marks, categories, rotation=45)
plt.yticks(tick_marks, categories)

# Menampilkan nilai di dalam confusion matrix
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Pengujian gambar baru
test_image_path = 'path/to/your/test/image.jpg'  # Ganti dengan path gambar uji Anda
predicted_category = classify_image(test_image_path, knn, target_size=(64, 64))
print(f"The predicted category for the test image is: {predicted_category}")
