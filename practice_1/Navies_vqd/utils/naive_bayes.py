import numpy as np

class NaiveBayesClassifierFromScratch:
    """
    Bộ phân loại Multinomial Naive Bayes viết từ đầu (Đã tối giản và tối ưu hóa vector).
    """
    def __init__(self, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

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
        # Chuyển đổi dữ liệu sang numpy array (hỗ trợ cả ma trận thưa tf-idf)
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.array(X)
        y = np.array(y).flatten()
        
        n_samples, n_features = X_arr.shape
        self.classes_ = np.unique(y)
        
        # 1. Tính toán Log Priors: log P(C_k) = log( số_mẫu_nhãn_c / tổng_số_mẫu )
        class_counts = np.array([np.sum(y == c) for c in self.classes_])
        self.class_log_prior_ = np.log(class_counts / n_samples)
        
        # 2. Tính toán Log Likelihoods: log P(w_i | C_k) với làm mịn Laplace
        feature_prob = []
        for c in self.classes_:
            X_c = X_arr[y == c]
            word_counts = X_c.sum(axis=0)  # Tổng tần suất của từng từ trong lớp c
            total_words = X_c.sum()         # Tổng số từ trong lớp c
            
            # Laplace smoothing: (counts + alpha) / (total + alpha * vocab_size)
            prob_c = (word_counts + self.alpha) / (total_words + self.alpha * n_features)
            feature_prob.append(prob_c)
            
        self.feature_log_prob_ = np.log(np.array(feature_prob))
        return self

    def predict(self, X):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.array(X)
        # Vectorization: log P(C|X) = log P(C) + X . log P(w|C)^T
        log_posteriors = X_arr.dot(self.feature_log_prob_.T) + self.class_log_prior_
        return self.classes_[np.argmax(log_posteriors, axis=1)]

    def predict_proba(self, X):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.array(X)
        log_posteriors = X_arr.dot(self.feature_log_prob_.T) + self.class_log_prior_
        
        # Softmax ổn định số học trên mảng 2D
        log_posteriors -= np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors)
        return exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)
