import numpy as np
from dataclasses import dataclass
class LinearRegressionScratch:
    """Mô hình Linear Regression tự triển khai sử dụng Gradient Descent."""

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.weights       = None
        self.bias          = None
        self.loss_history  = []

    def get_params(self, deep=True):
        return {'learning_rate': self.learning_rate, 'epochs': self.epochs}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # Bước 1: Khởi tạo weights = 0, bias = 0
        self.weights      = np.zeros(n_features)
        self.bias         = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            # Bước 2: Forward pass
            y_pred = np.dot(X, self.weights) + self.bias

            # Bước 3: Tính gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Bước 4: Cập nhật tham số (Gradient Descent)
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # Bước 5: Ghi lại loss
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, X) -> np.ndarray:
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias

@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    value: float | None = None

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DTR:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 3, min_samples_leaf: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: Node | None = None

    def __mse(self, y_actual: np.ndarray, y_hat: float) -> float:
        """Tính Mean Squared Error giữa y thực tế và một giá trị dự đoán.

        Args:
            y_actual (np.ndarray): Mảng target thực tế.
            y_hat (float): Giá trị dự đoán đại diện cho node.

        Returns:
            float: Giá trị MSE.
        """
        residual = y_actual - y_hat
        return float(np.mean(residual ** 2))

    def set_params(self, **params) -> "DTR":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DTR":
        """Train model Decision Tree Regressor với tập input là dữ liệu dạng NumPy.

        Args:
            X (np.ndarray): Ma trận feature có shape `(n_samples, n_features)`.
            y (np.ndarray): Vector target có shape `(n_samples,)`.

        Returns:
            DTR: Mô hình sau khi đã xây cây.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.root = self.__build_tree(X, y, 0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán target cho dữ liệu đầu vào dạng NumPy.

        Args:
            X (np.ndarray): Ma trận feature có shape `(n_samples, n_features)`.

        Returns:
            np.ndarray: Vector dự đoán có shape `(n_samples,)`.
        """
        X = np.asarray(X)
        return np.array([self.__traverse_tree(row, self.root) for row in X])

    def __traverse_tree(self, x: np.ndarray, node: Node) -> float:
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.__traverse_tree(x, node.left)

        return self.__traverse_tree(x, node.right)

    def __build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples = X.shape[0]

        if (depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_samples < 2 * self.min_samples_leaf
            or len(np.unique(y)) == 1):
            return Node(value=float(np.mean(y)))

        best_feature, best_threshold = self._get_best_split_criteria(X, y)

        if best_feature is None:
            return Node(value=float(np.mean(y)))

        left_mask = X[:, best_feature] <= best_threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[~left_mask]
        y_right = y[~left_mask]

        left_child = self.__build_tree(X_left, y_left, depth + 1)
        right_child = self.__build_tree(X_right, y_right, depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            value=None,
        )

    def _get_best_split_criteria(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float] | tuple[None, None]:
        """Tìm feature index và threshold tốt nhất để chia node hiện tại.
        (Feature index tức là ta sẽ thu thập index của best feature đó)
        Args:
            X (np.ndarray): Ma trận feature của node hiện tại, shape
                `(n_samples, n_features)`.
            y (np.ndarray): Vector target của node hiện tại, shape `(n_samples,)`.

        Returns:
            tuple[int, float] | tuple[None, None]: Cặp `(feature_index, threshold)`
            tốt nhất. Nếu không tìm được split hợp lệ, trả về `(None, None)`.

        Thuật toán chính:
            1. Tính MSE của node hiện tại làm mốc lỗi ban đầu.
            2. Duyệt qua từng feature index trong X.
            3. Với mỗi feature, tạo candidate threshold bằng midpoint giữa các
               giá trị unique liên tiếp sau khi sắp xếp.
            4. Với mỗi threshold, chia y thành nhánh trái/phải bằng boolean mask.
            5. Bỏ qua split nếu một nhánh có ít mẫu hơn `min_samples_leaf`.
            6. Tính weighted MSE của split và cập nhật split tốt nhất nếu lỗi nhỏ hơn.
        """
        current_mse = self.__mse(y, float(np.mean(y)))
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]

        for feature in range(n_features):
            x_arr = X[:, feature]
            unique_vals = np.sort(np.unique(x_arr))
            splits = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for split in splits:
                left_mask = x_arr <= split

                n_left = np.sum(left_mask)
                n_right = len(y) - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]

                left_mse = self.__mse(y_left, float(np.mean(y_left)))
                right_mse = self.__mse(y_right, float(np.mean(y_right)))
                weighted_mse = ((left_mse * n_left) + (right_mse * n_right)) / len(y)

                if weighted_mse < current_mse:
                    current_mse = weighted_mse
                    best_feature = feature
                    best_threshold = float(split)

        return (best_feature, best_threshold)

class RFR:
    """
    Random Forest Regressor tự triển khai.
    Sử dụng Bootstrap Sampling và Feature Randomness để xây dựng
    một ensemble các Decision Tree Regressor độc lập.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 5,
                 min_samples_split: int = 3, min_samples_leaf: int = 2,
                 max_features: int | None = None, random_state: int | None = None):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.random_state      = random_state
        self.trees_: list[DTR] = []

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        """Huấn luyện Random Forest bằng Bootstrap Aggregating (Bagging)."""
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # Tự động chọn max_features nếu không chỉ định (~sqrt hoặc n//3 cho regression)
        max_features = self.max_features or max(1, n_features // 3)

        rng = np.random.RandomState(self.random_state)
        self.trees_ = []

        for i in range(self.n_estimators):
            # Bootstrap Sampling: lấy mẫu có hoàn lại
            bootstrap_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[bootstrap_idx]
            y_boot = y[bootstrap_idx]

            # Tạo và huấn luyện cây con
            tree = DTR(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=int(rng.randint(0, 2**31))
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        return self

    def predict(self, X) -> np.ndarray:
        """Dự đoán bằng cách lấy trung bình kết quả của tất cả cây."""
        X = np.array(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(all_preds, axis=0)

class LeafNode:
    def __init__(self, sample_indices, depth, weight, score):
        self.sample_indices = sample_indices
        self.depth = depth
        self.weight = weight
        self.score = score
        self.is_leaf = True
        
        self.feature_idx = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.gain = -1.0

class LeafWiseTreeRegressor:
    def __init__(self, max_depth=3, max_leaf_nodes=8, l2_reg=1.0, min_child_samples=5):
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_reg = l2_reg
        self.min_child_samples = min_child_samples
        self.root = None
        self.leaves = []

    def _find_best_split(self, X_binned, gradients, sample_indices, n_bins):
        n_samples = len(sample_indices)
        if n_samples <= 2 * self.min_child_samples:
            return None, None, -1.0
            
        G_total = np.sum(gradients[sample_indices])
        H_total = n_samples
        
        best_gain = -1.0
        best_feat = None
        best_thresh = None
        
        X_sub = X_binned[sample_indices]
        grad_sub = gradients[sample_indices]
        
        n_features = X_binned.shape[1]
        for f in range(n_features):
            max_val = int(np.max(X_sub[:, f]))
            if max_val == 0:
                continue
            
            hist_H = np.bincount(X_sub[:, f])
            hist_G = np.bincount(X_sub[:, f], weights=grad_sub)
            
            if len(hist_G) < max_val + 1:
                hist_G = np.pad(hist_G, (0, max_val + 1 - len(hist_G)))
                hist_H = np.pad(hist_H, (0, max_val + 1 - len(hist_H)))
                
            G_L = 0.0
            H_L = 0
            
            for b in range(len(hist_G) - 1):
                G_L += hist_G[b]
                H_L += hist_H[b]
                G_R = G_total - G_L
                H_R = H_total - H_L
                
                if H_L >= self.min_child_samples and H_R >= self.min_child_samples:
                    gain = 0.5 * (
                        (G_L ** 2) / (H_L + self.l2_reg) +
                        (G_R ** 2) / (H_R + self.l2_reg) -
                        (G_total ** 2) / (H_total + self.l2_reg)
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = f
                        best_thresh = b
                        
        return best_feat, best_thresh, best_gain

    def fit(self, X_binned, gradients, n_bins):
        n_samples = X_binned.shape[0]
        G_total = np.sum(gradients)
        H_total = n_samples
        
        root_weight = G_total / (H_total + self.l2_reg)
        root_score = 0.5 * (G_total ** 2) / (H_total + self.l2_reg)
        
        self.root = LeafNode(np.arange(n_samples), depth=0, weight=root_weight, score=root_score)
        active_leaves = [self.root]
        
        feat, thresh, gain = self._find_best_split(X_binned, gradients, self.root.sample_indices, n_bins)
        self.root.feature_idx = feat
        self.root.threshold = thresh
        self.root.gain = gain
        
        leaf_count = 1
        
        while leaf_count < self.max_leaf_nodes:
            best_leaf_idx = -1
            best_gain = -1.0
            for i, leaf in enumerate(active_leaves):
                if leaf.gain > best_gain and leaf.depth < self.max_depth:
                    best_gain = leaf.gain
                    best_leaf_idx = i
            
            if best_leaf_idx == -1 or best_gain <= 0.0:
                break
                
            leaf_to_split = active_leaves.pop(best_leaf_idx)
            
            feature_idx = leaf_to_split.feature_idx
            threshold = leaf_to_split.threshold
            
            left_mask = X_binned[leaf_to_split.sample_indices, feature_idx] <= threshold
            left_indices = leaf_to_split.sample_indices[left_mask]
            right_indices = leaf_to_split.sample_indices[~left_mask]
            
            G_L = np.sum(gradients[left_indices])
            H_L = len(left_indices)
            left_weight = G_L / (H_L + self.l2_reg)
            left_score = 0.5 * (G_L ** 2) / (H_L + self.l2_reg)
            
            G_R = np.sum(gradients[right_indices])
            H_R = len(right_indices)
            right_weight = G_R / (H_R + self.l2_reg)
            right_score = 0.5 * (G_R ** 2) / (H_R + self.l2_reg)
            
            left_child = LeafNode(left_indices, depth=leaf_to_split.depth + 1, weight=left_weight, score=left_score)
            right_child = LeafNode(right_indices, depth=leaf_to_split.depth + 1, weight=right_weight, score=right_score)
            
            leaf_to_split.left_child = left_child
            leaf_to_split.right_child = right_child
            leaf_to_split.is_leaf = False
            
            f_L, t_L, g_L = self._find_best_split(X_binned, gradients, left_child.sample_indices, n_bins)
            left_child.feature_idx = f_L
            left_child.threshold = t_L
            left_child.gain = g_L
            
            f_R, t_R, g_R = self._find_best_split(X_binned, gradients, right_child.sample_indices, n_bins)
            right_child.feature_idx = f_R
            right_child.threshold = t_R
            right_child.gain = g_R
            
            active_leaves.append(left_child)
            active_leaves.append(right_child)
            
            leaf_count += 1
            
        self.leaves = self._collect_leaves(self.root)

    def _collect_leaves(self, node):
        if node.is_leaf:
            return [node]
        return self._collect_leaves(node.left_child) + self._collect_leaves(node.right_child)

    def predict(self, X_binned):
        preds = np.zeros(X_binned.shape[0])
        self._predict_recursive(self.root, X_binned, np.arange(X_binned.shape[0]), preds)
        return preds
        
    def _predict_recursive(self, node, X_binned, indices, preds):
        if len(indices) == 0:
            return
        if node.is_leaf:
            preds[indices] = node.weight
            return
        left_mask = X_binned[indices, node.feature_idx] <= node.threshold
        self._predict_recursive(node.left_child, X_binned, indices[left_mask], preds)
        self._predict_recursive(node.right_child, X_binned, indices[~left_mask], preds)

class LightGBMFromScratch:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=5, max_leaf_nodes=15, n_bins=32, l2_reg=1.0, min_child_samples=5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.n_bins = n_bins
        self.l2_reg = l2_reg
        self.min_child_samples = min_child_samples
        self.base_pred = None
        self.trees = []
        self.bin_edges = {}

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "n_bins": self.n_bins,
            "l2_reg": self.l2_reg,
            "min_child_samples": self.min_child_samples
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _bin_features(self, X, fit=False):
        X_arr = np.array(X)
        X_binned = np.zeros(X_arr.shape, dtype=np.int32)
        for col in range(X_arr.shape[1]):
            if fit:
                unique_vals = np.unique(X_arr[:, col])
                if len(unique_vals) <= self.n_bins:
                    self.bin_edges[col] = ('unique', unique_vals)
                else:
                    percentiles = np.linspace(0, 100, self.n_bins + 1)
                    edges = np.percentile(X_arr[:, col], percentiles)
                    edges = np.unique(edges)
                    self.bin_edges[col] = ('edges', edges)
            
            bin_type, edges = self.bin_edges[col]
            if bin_type == 'unique':
                X_binned[:, col] = np.searchsorted(edges, X_arr[:, col])
            else:
                X_binned[:, col] = np.digitize(X_arr[:, col], edges[1:-1])
        return X_binned

    def fit(self, X, y):
        if hasattr(y, 'values'):
            y_arr = y.values.flatten()
        else:
            y_arr = np.array(y).flatten()
            
        X_binned = self._bin_features(X, fit=True)
        self.base_pred = np.mean(y_arr)
        f_m = np.full(len(y_arr), self.base_pred, dtype=np.float64)
        
        self.trees = []
        for i in range(self.n_estimators):
            gradient = y_arr - f_m
            tree = LeafWiseTreeRegressor(
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                l2_reg=self.l2_reg,
                min_child_samples=self.min_child_samples
            )
            tree.fit(X_binned, gradient, self.n_bins)
            f_m += self.learning_rate * tree.predict(X_binned)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X_binned = self._bin_features(X, fit=False)
        preds = np.full(X.shape[0], self.base_pred, dtype=np.float64)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X_binned)
        return preds