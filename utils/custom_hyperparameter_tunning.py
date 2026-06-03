import numpy as np
import itertools

class CustomGridSearchCV:
    def __init__(self, estimator, param_grid, cv):
        """Khởi tạo bộ tìm siêu tham số

        Args:
            estimator (_type_): Model
            param_grid (_type_): Dictionary chứa các tham số cần thử
            cv (_type_): Đối tượng K-Fold
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        
        self.best_params_ = None
        self.best_score_ = -np.inf 
        self.best_estimator_ = None
        self.cv_results_ = []      

    def _calculate_r2(self, y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    def fit(self, X, y):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"Bắt đầu GridSearchCV: Sẽ chạy {len(combinations)} tổ hợp tham số, mỗi tổ hợp {self.cv.n_splits} folds.")
        print(f"Tổng cộng số lần huấn luyện: {len(combinations) * self.cv.n_splits} lần.\n")

        for idx, params in enumerate(combinations):
            self.estimator.set_params(**params)
            
            fold_scores = []
            
            for train_idx, val_idx in self.cv.split(X):
                X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
                
                self.estimator.fit(X_train, y_train)
                y_pred = self.estimator.predict(X_val)
                
                score = self._calculate_r2(y_val, y_pred)
                fold_scores.append(score)
            
            mean_score = np.mean(fold_scores)
            
            self.cv_results_.append({
                'params': params,
                'mean_test_score': mean_score,
                'fold_scores': fold_scores
            })
            
            print(f"[{idx+1}/{len(combinations)}] Tham số: {params} --> R2 trung bình: {mean_score:.4f}")
            
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
        
        print(f"Tham số TỐT NHẤT: {self.best_params_}")
        print(f"R2 CV TỐT NHẤT  : {self.best_score_:.4f}")
        
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y)
        self.best_estimator_ = self.estimator
        
        return self