import numpy as np
from collections import Counter

# Load dữ liệu Iris (ở đây là dữ liệu giả định)
# Trong trường hợp bạn có dữ liệu, hãy đặt dữ liệu vào các biến `X` và `y_true`.
# Ví dụ cho bộ dữ liệu Iris:
from sklearn.datasets import load_iris
iris = load_iris()
X, y_true = iris.data, iris.target

# Hàm tính khoảng cách Euclidean    
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Hàm khởi tạo các centroids ngẫu nhiên
def initialize_centroids(X, k):
    np.random.seed(42)  # Đảm bảo kết quả nhất quán
    random_indices = np.random.choice(len(X), k, replace=False)
    return X[random_indices]

# Hàm gán mỗi điểm dữ liệu vào cụm gần nhất
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters.append(closest_centroid)
    return clusters

# Hàm cập nhật vị trí của các centroids
def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = [X[j] for j in range(len(X)) if clusters[j] == i]
        if cluster_points:  # Kiểm tra nếu cụm không rỗng
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:  # Nếu cụm rỗng, giữ nguyên centroid cũ
            new_centroids.append(np.zeros(X.shape[1]))
    return new_centroids

# Thuật toán K-means chính
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Áp dụng thuật toán K-means với k=3
k = 5
clusters, centroids = kmeans(X, k)

# Hàm tính F1-score thủ công
def calculate_f1_score(y_true, y_pred):
    true_labels = Counter(y_true)
    pred_labels = Counter(y_pred)
    tp = sum((true_labels & pred_labels).values())
    fp = len(y_pred) - tp
    fn = len(y_true) - tp
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return f1

# Hàm tính Rand Index thủ công
def calculate_rand_index(y_true, y_pred):
    tp_plus_fp = sum(Counter(y_pred).values())
    tp_plus_fn = sum(Counter(y_true).values())
    tp = sum((Counter(y_true) & Counter(y_pred)).values())
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = len(y_true) * (len(y_true) - 1) / 2 - tp - fp - fn
    rand_index = (tp + tn) / (tp + fp + fn + tn)
    return rand_index

# Đánh giá F1-score
f1_score_value = calculate_f1_score(y_true, clusters)
print(f"F1 Score: {f1_score_value:.2f}")

# Đánh giá Rand Index
rand_index_value = calculate_rand_index(y_true, clusters)
print(f"Rand Index: {rand_index_value:.2f}")

# Đánh giá NMI (Normalized Mutual Information) và DB (Davies-Bouldin) có thể rất phức tạp để tính thủ công.
# Để tính NMI, bạn cần thực hiện tính toán entropy của từng cụm và các nhãn thực, sau đó tính MI (mutual information).
# Để tính DB Index, cần đo khoảng cách giữa các cụm và độ phân tán trong từng cụm.

# Nếu cần thiết, bạn có thể gọi thư viện sklearn để xác thực các giá trị F1-score và Rand Index tính tay.
