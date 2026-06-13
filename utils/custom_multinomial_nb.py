import numpy as np
from scipy.sparse import issparse


class CustomMultinomialNB:
    """
    Custom Multinomial Naive Bayes

    Hỗ trợ:
    - fit()
    - predict()
    - predict_proba()
    - get_params()
    - set_params()

    Tương thích:
    - CustomGridSearchCV
    - CustomKFold
    """

    def __init__(self, alpha=1.0, fit_prior=True):

        self.alpha = alpha
        self.fit_prior = fit_prior

        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    # ==================================================
    # Các hàm bắt buộc để GridSearchCV custom hoạt động
    # ==================================================

    def get_params(self, deep=True):

        return {
            "alpha": self.alpha,
            "fit_prior": self.fit_prior
        }

    def set_params(self, **params):

        for key, value in params.items():
            setattr(self, key, value)

        return self

    # ==================================================
    # Huấn luyện mô hình
    # ==================================================

    def fit(self, X, y):

        y = np.array(y)

        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)

        self.feature_log_prob_ = np.zeros(
            (n_classes, n_features)
        )

        for idx, cls in enumerate(self.classes_):

            X_cls = X[y == cls]

            # -------------------------
            # Prior Probability
            # -------------------------

            if self.fit_prior:

                prior = X_cls.shape[0] / X.shape[0]

            else:

                prior = 1 / n_classes

            self.class_log_prior_[idx] = np.log(prior)

            # -------------------------
            # Đếm số lần xuất hiện
            # của feature
            # -------------------------

            if issparse(X_cls):

                feature_count = np.asarray(
                    X_cls.sum(axis=0)
                ).flatten()

            else:

                feature_count = X_cls.sum(axis=0)

            # -------------------------
            # Laplace Smoothing
            # -------------------------

            smoothed_fc = (
                feature_count + self.alpha
            )

            smoothed_total = (
                np.sum(feature_count)
                + self.alpha * n_features
            )

            self.feature_log_prob_[idx] = np.log(
                smoothed_fc / smoothed_total
            )

        return self

    # ==================================================
    # Joint Log Likelihood
    # ==================================================

    def _joint_log_likelihood(self, X):

        if issparse(X):

            return (
                X @ self.feature_log_prob_.T
            ) + self.class_log_prior_

        return (
            np.dot(
                X,
                self.feature_log_prob_.T
            )
        ) + self.class_log_prior_

    # ==================================================
    # Predict
    # ==================================================

    def predict(self, X):

        jll = self._joint_log_likelihood(X)

        indices = np.argmax(
            jll,
            axis=1
        )

        return self.classes_[indices]

    # ==================================================
    # Predict Probability
    # ==================================================

    def predict_proba(self, X):

        jll = self._joint_log_likelihood(X)

        max_log = np.max(
            jll,
            axis=1,
            keepdims=True
        )

        exp_prob = np.exp(
            jll - max_log
        )

        probs = exp_prob / np.sum(
            exp_prob,
            axis=1,
            keepdims=True
        )

        return probs