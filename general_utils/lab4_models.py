import numpy as np
from dataclasses import dataclass

class MLPScratch:
    """Mô hình MLP custom 
    """

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
