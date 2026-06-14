import os
import sys
import joblib
import numpy as np
import time
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ----------------- CODE MÔ HÌNH FROM SCRATCH (TRÍCH XUẤT TỪ NOTEBOOK) -----------------

import sys
import numpy as np

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

sys.modules['__main__'].LightGBMFromScratch = LightGBMFromScratch


# ----------------- KỊCH BẢN THÍ NGHIỆM SO SÁNH n_bins -----------------

if __name__ == '__main__':
    # Xác định đường dẫn repo root và data directory
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(REPO_ROOT, 'practice_2', 'data', 'ready_for_train')
    
    # Load data
    print("Loading data from:", DATA_DIR)
    X_train = joblib.load(os.path.join(DATA_DIR, 'X_train_final.pkl'))
    X_test = joblib.load(os.path.join(DATA_DIR, 'X_test_final.pkl'))
    y_train = joblib.load(os.path.join(DATA_DIR, 'y_train_log.pkl'))
    y_test = joblib.load(os.path.join(DATA_DIR, 'y_test_log.pkl'))
    
    if hasattr(y_train, 'values'):
        y_train = y_train.values.flatten()
    else:
        y_train = np.array(y_train).flatten()
        
    if hasattr(y_test, 'values'):
        y_test = y_test.values.flatten()
    else:
        y_test = np.array(y_test).flatten()
        
    y_test_original = np.expm1(y_test)
    
    print("Data shapes:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)
    
    # Cấu hình thử nghiệm
    bins_list = [16, 32, 64, 128, 255]
    results = []
    
    # Siêu tham số tối ưu thời gian chạy (estimators thấp để mô hình Scratch chạy nhanh)
    n_est = 20
    lr = 0.1
    max_d = 4
    max_l = 8
    
    print("=" * 70)
    print(f"BẮT ĐẦU CHẠY THÍ NGHIỆM SO SÁNH n_bins")
    print(f"Cấu hình: n_estimators={n_est}, learning_rate={lr}, max_depth={max_d}, max_leaves={max_l}")
    print("=" * 70)
    
    for nb in bins_list:
        print(f"\n--- Đang huấn luyện với n_bins = {nb} ---")
        
        # 1. From Scratch
        t0 = time.time()
        model_scratch = LightGBMFromScratch(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=max_d,
            max_leaf_nodes=max_l,
            n_bins=nb
        )
        model_scratch.fit(X_train, y_train)
        t_fit_scratch = time.time() - t0
        
        y_pred_log_scratch = model_scratch.predict(X_test)
        y_pred_scratch = np.expm1(y_pred_log_scratch)
        
        r2_log_scratch = r2_score(y_test, y_pred_log_scratch)
        r2_orig_scratch = r2_score(y_test_original, y_pred_scratch)
        
        # 2. Library
        t0 = time.time()
        model_lib = LGBMRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=max_d,
            num_leaves=max_l,
            max_bin=nb,
            min_child_samples=1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model_lib.fit(X_train, y_train)
        t_fit_lib = time.time() - t0
        
        y_pred_log_lib = model_lib.predict(X_test)
        y_pred_lib = np.expm1(y_pred_log_lib)
        
        r2_log_lib = r2_score(y_test, y_pred_log_lib)
        r2_orig_lib = r2_score(y_test_original, y_pred_lib)
        
        results.append({
            'n_bins': nb,
            'Scratch_FitTime(s)': t_fit_scratch,
            'Lib_FitTime(s)': t_fit_lib,
            'Scratch_R2(Log)': r2_log_scratch,
            'Lib_R2(Log)': r2_log_lib,
            'Scratch_R2(Orig)': r2_orig_scratch,
            'Lib_R2(Orig)': r2_orig_lib
        })
        
        print(f"  [From Scratch] Fit Time: {t_fit_scratch:.4f}s | R2 (Log): {r2_log_scratch:.4f} | R2 (Orig): {r2_orig_scratch:.4f}")
        print(f"  [Library]      Fit Time: {t_fit_lib:.4f}s | R2 (Log): {r2_log_lib:.4f} | R2 (Orig): {r2_orig_lib:.4f}")
        
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 90)
    print("BẢNG KẾT QUẢ THÍ NGHIỆM SO SÁNH HIỆU NĂNG THEO SỐ LƯỢNG GIỎ CHIA (n_bins)")
    print("=" * 90)
    print(df_results.to_string(index=False))
    print("=" * 90)
    
    # Save to CSV
    csv_out = os.path.join(os.path.dirname(__file__), 'bins_comparison_results.csv')
    df_results.to_csv(csv_out, index=False)
    print(f"Đã lưu kết quả chi tiết vào: {csv_out}")
