"""
Module này dùng để thử nhiều tổ hợp siêu tham số cho một mô hình,
đánh giá từng tổ hợp bằng cross validation, sau đó chọn ra bộ tham số
có điểm trung bình tốt nhất theo metric được chỉ định.
"""

import numpy as np
import itertools
from scipy.sparse import issparse

class CustomGridSearchCV:
    """Tìm kiếm siêu tham số tốt nhất cho mô hình bằng Grid Search.

    Attributes:
        best_params_ (dict | None): Bộ siêu tham số có điểm đánh giá tốt nhất.
        best_score_ (float): Điểm trung bình tốt nhất trên các fold.
        best_estimator_ (object | None): Mô hình đã được huấn luyện lại với
            `best_params_` trên toàn bộ dữ liệu.
        cv_results_ (list[dict]): Danh sách kết quả đánh giá của từng tổ hợp
            tham số, gồm `params` và các điểm dạng `mean_test_<metric>`.
    """

    def __init__(self, estimator, param_grid, cv, scoring='f1'):
        """Khởi tạo bộ tìm kiếm siêu tham số.

        Args:
            estimator (object): Mô hình cần tối ưu siêu tham số. Mô hình phải
                có các phương thức `set_params`, `fit` và `predict`.
            param_grid (dict): Tập các siêu tham số cần thử. Mỗi key là tên
                tham số của mô hình, mỗi value là danh sách các giá trị có thể
                thử cho tham số đó.
            cv (object): Bộ chia cross validation. Đối tượng này cần có thuộc
                tính `n_splits` và phương thức `split(X, y)` trả về các cặp
                chỉ số train/validation.
            scoring (str hoặc list[str]): Metric dùng để đánh giá mô hình.
                Nếu truyền list, metric đầu tiên sẽ được dùng để chọn
                `best_params_` và `best_score_`. Các giá trị đang hỗ trợ gồm
                `neg_mse`, `neg_rmse`, `r2` đối với Regression, `accuracy` và
                `f1` đối với Classification.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.scoring_list_ = [scoring] if isinstance(scoring, str) else list(scoring)
        if not self.scoring_list_:
            raise ValueError("scoring phải là str hoặc list[str] không rỗng")
        self.refit_metric_ = self.scoring_list_[0]
        
        self.best_params_ = None
        self.best_score_ = -np.inf 
        self.best_estimator_ = None
        self.cv_results_ = []      

    def _score(self, y_true, y_pred, metric):
        """Tính điểm đánh giá dựa trên metric đã chọn.

        Args:
            y_true (array-like): Nhãn hoặc giá trị thực tế của tập validation.
            y_pred (array-like): Nhãn hoặc giá trị do mô hình dự đoán.
            metric (str): Tên metric cần tính.

        Returns:
            float: Điểm đánh giá tương ứng với `metric`.
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        

        # Regression metrics
        if metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        elif metric == 'neg_mse':
            return -np.mean((y_true - y_pred) ** 2)

        elif metric == 'neg_rmse':
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Classifications metrics

        elif metric == 'accuracy':
            return np.mean(y_true == y_pred)
            
        elif metric == 'f1':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            # F1 = 2*TP / (2*TP + FP + FN)
            denominator = (2 * tp + fp + fn)
            return (2 * tp) / denominator if denominator > 0 else 0.0
            
        raise ValueError(f"Metric không được hỗ trợ: {metric}")

    def fit(self, X, y):
        """Thực hiện Grid Search và huấn luyện mô hình tốt nhất.

        Args:
            X : Dữ liệu
                đầu vào dùng để huấn luyện và đánh giá mô hình.
            y : Nhãn hoặc giá trị mục tiêu tương
                ứng với từng mẫu trong `X`.

        Returns:
            CustomGridSearchCV: Chính đối tượng hiện tại sau khi đã tìm được
            tham số tốt nhất và huấn luyện `best_estimator_`.
        """
        is_sparse = issparse(X)
        y = np.array(y) if not hasattr(y, 'iloc') else y
        
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"Bắt đầu GridSearchCV: {len(combinations)} tổ hợp tham số, {self.cv.n_splits} folds.")
        
        for idx, params in enumerate(combinations):
            self.estimator.set_params(**params)
            fold_scores = {metric: [] for metric in self.scoring_list_}
            
            for train_idx, val_idx in self.cv.split(X, y):
                if is_sparse:
                    X_train, X_val = X[train_idx], X[val_idx]
                else:
                    X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                    X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                
                y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
                
                self.estimator.fit(X_train, y_train)
                y_pred = self.estimator.predict(X_val)
                for metric in self.scoring_list_:
                    score = self._score(y_val, y_pred, metric)
                    fold_scores[metric].append(score)
            
            mean_scores = {
                metric: np.mean(scores)
                for metric, scores in fold_scores.items()
            }
            result = {'params': params}
            result.update({
                f'mean_test_{metric}': score
                for metric, score in mean_scores.items()
            })
            self.cv_results_.append(result)
            
            score_text = ', '.join(
                f"{metric}: {score:.4f}"
                for metric, score in mean_scores.items()
            )
            print(f"[{idx+1}/{len(combinations)}] Params: {params} --> {score_text}")
            
            refit_score = mean_scores[self.refit_metric_]
            if refit_score > self.best_score_:
                self.best_score_ = refit_score
                self.best_params_ = params
        
        print(f"\n-> Tham số TỐT NHẤT: {self.best_params_}")
        print(f"-> Điểm {self.refit_metric_} TỐT NHẤT: {self.best_score_:.4f}")
        
        # Train model on all Training set
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y)
        self.best_estimator_ = self.estimator
        return self
