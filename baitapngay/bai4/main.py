import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Đường dẫn đến thư mục chứa ảnh
image_folder_path = './bai4/data'  # Cập nhật đường dẫn tới thư mục ảnh
output_folder_path
# Chuẩn bị danh sách để lưu ảnh và nhãn
data = []
labels = []

# Đọc từng ảnh trong thư mục
for filename in os.listdir(image_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Định dạng ảnh có thể khác nhau
        # Đọc ảnh
        img_path = os.path.join(image_folder_path, filename)
        img = cv2.imread(img_path)
        
        # Resize ảnh về kích thước chuẩn (ví dụ: 100x100)
        img = cv2.resize(img, (64, 128))  # Điều chỉnh kích thước tùy theo yêu cầu

        # Chuyển ảnh sang dạng mảng và thêm vào data
        data.append(img.flatten())  # Dùng .flatten() để chuyển ảnh thành vector nếu cần

        # Lấy nhãn từ tên tệp hoặc đặt nhãn mặc định (ví dụ: filename chứa nhãn ở phần đầu)
        label = filename.split('_')[0]  # Giả sử nhãn là phần đầu của tên tệp trước dấu '_'
        labels.append(label)

# Chuyển data và labels thành dạng numpy array
data = np.array(data)
labels = np.array(labels)

# Các tỷ lệ chia train-test
ratios = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.4, 0.6)]

# Kết quả đánh giá
results = []

for train_size, test_size in ratios:
    print(f"\n\nKịch bản chia dữ liệu: Train={int(train_size*100)}%, Test={int(test_size*100)}%")

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size, random_state=42)

    # Huấn luyện và đánh giá SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("SVM - Accuracy:", accuracy_svm)
    print("SVM - Classification Report:\n", classification_report(y_test, y_pred_svm, zero_division=0))

    # Huấn luyện và đánh giá KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("KNN - Accuracy:", accuracy_knn)
    print("KNN - Classification Report:\n", classification_report(y_test, y_pred_knn, zero_division=0))

    # Lưu kết quả
    results.append({
        "train_size": train_size,
        "test_size": test_size,
        "accuracy_svm": accuracy_svm,
        "accuracy_knn": accuracy_knn
    })


# So sánh kết quả
print("\nTóm tắt kết quả phân lớp:")
for res in results:
    print(f"Train={int(res['train_size']*100)}%, Test={int(res['test_size']*100)}% -> SVM: {res['accuracy_svm']:.2f}, KNN: {res['accuracy_knn']:.2f}")
