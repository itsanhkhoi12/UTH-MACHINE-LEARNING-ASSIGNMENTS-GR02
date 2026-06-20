import numpy as np
from scipy.spatial.distance import cdist

class HierarchicalClusteringScratch:
    def __init__(self, n_clusters=4, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.clusters_ = None
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        # Mỗi điểm dữ liệu ban đầu là một cụm riêng lẻ
        self.clusters_ = {i: [i] for i in range(n_samples)}
        
        # Tính toán ma trận khoảng cách Euclidean pairwise ban đầu giữa các điểm
        point_dist = cdist(X, X, metric='euclidean')
        
        # Danh sách các cụm hiện đang hoạt động
        active_clusters = list(range(n_samples))
        
        # Bản sao của ma trận khoảng cách phục vụ việc gộp cụm
        # Gán đường chéo bằng vô cùng để tránh tự gộp với chính mình
        cluster_distances = point_dist.copy()
        np.fill_diagonal(cluster_distances, np.inf)
        
        # Định nghĩa các hàm tính khoảng cách linkage giữa hai cụm
        def get_cluster_dist(c1_indices, c2_indices):
            if self.linkage == 'centroid':
                centroid1 = np.mean(X[c1_indices], axis=0)
                centroid2 = np.mean(X[c2_indices], axis=0)
                return np.linalg.norm(centroid1 - centroid2)
            elif self.linkage == 'ward':
                n1 = len(c1_indices)
                n2 = len(c2_indices)
                centroid1 = np.mean(X[c1_indices], axis=0)
                centroid2 = np.mean(X[c2_indices], axis=0)
                return np.sqrt((2.0 * n1 * n2) / (n1 + n2)) * np.linalg.norm(centroid1 - centroid2)
            
            sub_matrix = point_dist[np.ix_(c1_indices, c2_indices)]
            if self.linkage == 'single':
                return np.min(sub_matrix)
            elif self.linkage == 'complete':
                return np.max(sub_matrix)
            elif self.linkage == 'average':
                return np.mean(sub_matrix)
            else:
                raise ValueError("Không hỗ trợ linkage này")
        
        # Thực hiện vòng lặp gộp cụm cho tới khi số cụm bằng n_clusters
        while len(active_clusters) > self.n_clusters:
            # Tìm hai cụm có khoảng cách nhỏ nhất
            min_idx = np.argmin(cluster_distances)
            c1, c2 = np.unravel_index(min_idx, cluster_distances.shape)
            
            # Gộp cụm c2 vào cụm c1
            self.clusters_[c1].extend(self.clusters_[c2])
            del self.clusters_[c2]
            
            # Hủy kích hoạt cụm c2
            active_clusters.remove(c2)
            cluster_distances[c2, :] = np.inf
            cluster_distances[:, c2] = np.inf
            
            # Cập nhật lại khoảng cách từ cụm mới c1 tới toàn bộ các cụm còn lại đang hoạt động
            c1_indices = self.clusters_[c1]
            for other in active_clusters:
                if other == c1:
                    continue
                other_indices = self.clusters_[other]
                dist = get_cluster_dist(c1_indices, other_indices)
                cluster_distances[c1, other] = dist
                cluster_distances[other, c1] = dist
                
        # Gán nhãn cụm cho từng điểm dữ liệu
        labels = np.zeros(n_samples, dtype=int)
        for label_idx, (cluster_id, indices) in enumerate(self.clusters_.items()):
            for idx in indices:
                labels[idx] = label_idx
                
        self.labels_ = labels
        return labels
