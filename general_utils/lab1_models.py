import pandas as pd
import numpy as np
import os, sys, time
from scipy.sparse import issparse
from collections import Counter

class LogisticRegressionScratch:
    """Triển khai Logistic Regression không dùng thư viện sklearn."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs
        }
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def sigmoid(self, z):
        # Áp dụng hàm kích hoạt Sigmoid đưa đầu ra tuyến tính về khoảng (0, 1)
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        # Tính toán Binary Cross-Entropy Loss (Log Loss)
        eps = 1e-15
        # Giới hạn giá trị của y_pred để tránh lỗi tràn số log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        y = np.array(y).flatten()
        n_samples, n_features = X.shape

        # Khởi tạo tham số trọng số (weights) và bias bằng 0
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        # Vòng lặp tối ưu hóa Gradient Descent
        for epoch in range(self.epochs):
            # Tính giá trị kết hợp tuyến tính z và xác suất dự đoán a
            linear = X.dot(self.weights) + self.bias
            pred = self.sigmoid(linear)

            # Tính toán đạo hàm riêng của Loss theo weights và bias
            dw = (1 / n_samples) * X.T.dot(pred - y)
            db = (1 / n_samples) * np.sum(pred - y)

            # Cập nhật tham số theo learning rate
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Lưu lại loss lịch sử để theo dõi độ hội tụ
            loss = self.compute_loss(y, pred)
            self.loss_history.append(loss)
            
        return self

    def predict_proba(self, X):
        # Trả về xác suất thuộc về lớp dương (lớp 1)
        linear = X.dot(self.weights) + self.bias
        return self.sigmoid(linear)

    def predict(self, X, threshold=0.5):
        # Dự đoán nhãn lớp (0 hoặc 1) dựa vào ngưỡng threshold chỉ định
        return (self.predict_proba(X) >= threshold).astype(int)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       
        self.threshold = threshold   
        self.left = left             
        self.right = right           
        self.value = value           
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeScratch:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
    
    def get_params(self, deep=True):
        return {
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "n_features": self.n_features
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        y = np.array(y).flatten()
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if n_samples == 0:
            return Node(value=0)

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            return Node(value=self._most_common_label(y))

        X_column = X[:, best_feature]
        if issparse(X_column):
            X_column = X_column.toarray().flatten()

        left_idxs, right_idxs = self._split(X_column, best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            
            if issparse(X_column):
                X_column = X_column.toarray().flatten()
            
            percentiles = np.percentile(X_column, [20, 40, 60, 80])
            thresholds = np.unique(percentiles) 

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
                    
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return -1 
        
        n = len(y)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (len(left_idxs) / n) * e_l + (len(right_idxs) / n) * e_r

        gain = parent_entropy - child_entropy
        return gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            x_row = X[i].toarray().flatten() if issparse(X) else X[i]
            preds.append(self._traverse_tree(x_row, self.root))
        return np.array(preds)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForestClassifierScratch:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        if hasattr(y, "values"):
            y = y.values
        else:
            y = np.array(y).flatten()
            
        self.trees = []
        for _ in range(self.n_estimators):
            X_samp, y_samp = self._bootstrap_samples(X, y)
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=int(np.sqrt(X.shape[1]))
            )
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)
            
        return self

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    # def predict(self, X):
    #     # Collect predictions from all tree
    #     tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
    #     tree_preds = np.swapaxes(tree_preds, 0, 1)
        
    #     # Majority Voting
    #     return np.array([Counter(pred).most_common(1)[0][0] for pred in tree_preds])
    def predict_proba(self, X):
        if hasattr(X, "toarray"): 
            X = X.toarray()
        elif hasattr(X, "values"):
            X = X.values
        else:
            X = np.array(X)
            
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        spam_votes = np.sum(tree_preds == 1, axis=1)
        spam_probabilities = spam_votes / self.n_estimators
        
        return spam_probabilities

    def predict(self, X, threshold=0.5):
        spam_probs = self.predict_proba(X)
        return (spam_probs >= threshold).astype(int)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class LinearSVMScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=100):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "lambda_param": self.lambda_param,
            "n_iters": self.n_iters
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        
        y = np.array(y).flatten()
        n_samples, n_features = X.shape
        
        # Conver label to 1 and -1 for SVM
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            margins = y_ * (X.dot(self.w) - self.b)
            
            misclassified = margins < 1
            
            y_mis = y_ * misclassified
            
            dw = 2 * self.lambda_param * self.w - (X.T.dot(y_mis) / n_samples)
            db = np.sum(y_mis) / n_samples
            
            # Update weight
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
        return self

    def predict(self, X, threshold=0.0):
        approx = X.dot(self.w) - self.b
        # return np.where(approx < 0, 0, 1)
        return np.where(approx < threshold, 0, 1)

class NaiveBayesClassifierFromScratch:
    """
    Bộ phân loại Multinomial Naive Bayes viết từ đầu.

    Parameters
    ----------
    alpha : float
        Hằng số làm mịn Laplace. Mặc định 1.0.
    force_alpha : bool
        Nếu False và alpha < 1e-10, alpha sẽ bị cắt về 1e-10 để tránh
        lỗi chia cho 0. Mặc định True (giữ nguyên alpha thiết lập).
    fit_prior : bool
        Nếu True, tính xác suất tiên nghiệm của các lớp từ dữ liệu huấn luyện.
        Nếu False, sử dụng phân phối đều. Mặc định True.
    class_prior : array-like hoặc None
        Mảng xác suất tiên nghiệm cố định do người dùng tự định nghĩa.
        Khi được gán, sẽ ghi đè cả fit_prior và phân phối từ dữ liệu. Mặc định None.
    """

    def __init__(self, alpha=1.0, force_alpha=True,
                 fit_prior=True, class_prior=None):
        self.alpha        = alpha
        self.force_alpha  = force_alpha
        self.fit_prior    = fit_prior
        self.class_prior  = class_prior
        self.classes_          = None
        self.class_priors_     = {}
        self.word_likelihoods_ = {}
        self.vocab_size_       = 0

    def get_params(self, deep=True):
        return {
            "alpha": self.alpha,
            "force_alpha": self.force_alpha,
            "fit_prior": self.fit_prior,
            "class_prior": self.class_prior
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_    = np.unique(y)
        self.vocab_size_ = n_features

        # Xác định giá trị alpha áp dụng
        eff_alpha = self.alpha
        if not self.force_alpha and self.alpha < 1e-10:
            eff_alpha = 1e-10

        # Tính toán xác suất tiên nghiệm của các lớp
        if self.class_prior is not None:
            for idx, c in enumerate(self.classes_):
                self.class_priors_[c] = self.class_prior[idx]
        elif not self.fit_prior:
            uniform = 1.0 / len(self.classes_)
            for c in self.classes_:
                self.class_priors_[c] = uniform
        else:
            for c in self.classes_:
                self.class_priors_[c] = np.sum(y == c) / n_samples

        # Tính toán log-likelihood của các từ kèm làm mịn Laplace
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.array(X)
        for c in self.classes_:
            X_c    = X_arr[y == c]
            total  = X_c.sum()
            counts = X_c.sum(axis=0)
            self.word_likelihoods_[c] = (
                (counts + eff_alpha) /
                (total  + eff_alpha * self.vocab_size_)
            )
        return self

    def predict(self, X):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.array(X)
        preds = []
        for row in X_arr:
            scores = {
                c: np.log(self.class_priors_[c]) +
                   np.sum(row * np.log(self.word_likelihoods_[c]))
                for c in self.classes_
            }
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

    def predict_proba(self, X):
        """Tính xác suất tiên nghiệm sau khi quan sát dữ liệu (Softmax ổn định số học)."""
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.array(X)
        proba = []
        for row in X_arr:
            log_scores = np.array([
                np.log(self.class_priors_[c]) +
                np.sum(row * np.log(self.word_likelihoods_[c]))
                for c in self.classes_
            ])
            # Softmax ổn định số học
            log_scores -= log_scores.max()
            exp_s = np.exp(log_scores)
            proba.append(exp_s / exp_s.sum())
        return np.array(proba)