# Giải Pháp Hồi Quy Dự Đoán Doanh Thu Sản Phẩm (Thành viên: vqd)

Thư mục `complete_solution` chứa toàn bộ giải pháp cá nhân hoàn chỉnh và độc lập cho bài toán hồi quy dự đoán doanh thu bán hàng (Sales Revenue) sử dụng các mô hình máy học nhóm cây và Boosting (HistGradientBoosting, LightGBM, XGBoost, CatBoost).

---

## 📂 Danh Sách Các Tệp Tin & Thư Mục Con

### 1. 📓 [complete_solution_vi/](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/complete_solution/complete_solution_vi/) (Notebooks Tiếng Việt)
Bộ giải pháp cá nhân được cấu trúc hóa thành **8 bước notebook tiếng Việt** hoàn chỉnh giúp người đọc theo dõi dễ dàng:
- **`01_introduction.ipynb`**: Giới thiệu bài toán hồi quy, các độ đo MAE, MSE, RMSE, R-squared và cấu trúc thư mục.
- **`02_data_checks.ipynb`**: Kiểm tra dữ liệu (kiểu dữ liệu, giá trị khuyết thiếu, outliers).
- **`03_data_cleaning.ipynb`**: Làm sạch dữ liệu (loại bỏ trùng lặp, xử lý các giá trị bất hợp lý).
- **`04_eda.ipynb`**: Phân tích khám phá dữ liệu (phân phối doanh số, tương quan giữa các biến liên tục và biến phân loại).
- **`05_feature_engineering.ipynb`**: Trích chọn và mã hóa đặc trưng (One-Hot Encoding, Label Encoding, Scaling).
- **`06_model_training.ipynb`**: Huấn luyện mô hình Cây quyết định từ đầu (`DecisionTreeRegressorScratch`) và so sánh với Scikit-learn.
- **`07_evaluation.ipynb`**: Đánh giá chi tiết các mô hình trên tập kiểm thử (vẽ đồ thị so sánh MAE, RMSE, R²).
- **`08_conclusion.ipynb`**: Tổng kết kết quả và đề xuất cải tiến.

### 2. 📓 [04_Model_Benchmark.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/complete_solution/04_Model_Benchmark.ipynb) (Đánh giá các thuật toán Boosting)
So sánh hiệu năng của 4 mô hình Gradient Boosting mạnh mẽ:
- HistGradientBoosting (Scikit-learn)
- LightGBM Regressor (Thư viện LightGBM)
- CatBoost Regressor (Thư viện CatBoost)
- XGBoost Regressor (Thư viện XGBoost)

### 3. 🐍 [compare_bins_experiment.py](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/complete_solution/compare_bins_experiment.py) & [bins_comparison_results.csv](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/complete_solution/bins_comparison_results.csv)
Script chạy thực nghiệm để so sánh các chiến lược phân nhóm (binning) khác nhau cho biến số liên tục (ví dụ: chia theo khoảng đều nhau, chia theo phân vị) nhằm nâng cao độ chính xác và giảm nhiễu cho mô hình học máy.

### 4. 📂 [LightGBM_vqd/](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/complete_solution/LightGBM_vqd/)
Mô hình LightGBM cá nhân được cấu hình tối ưu của `vqd`.

---

## 📊 Kết Quả Thực Nghiệm & Đánh Giá

Bảng so sánh hiệu năng các mô hình trên tập dữ liệu kiểm thử (Test Set):

| Mô hình (Model) | MAE | RMSE | R² Score | Thời gian Fit (ms) | Thời gian Dự đoán (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Sklearn)** | 1751.37 | 3072.06 | **0.4850** | 932.54 | 19.75 |
| **LightGBM Regressor** | **1748.12** | 3078.36 | 0.4829 | **347.71** | 12.51 |
| **CatBoost Regressor** | 1772.32 | 3117.80 | 0.4695 | 423.65 | **2.66** |
| **XGBoost Regressor** | 1787.86 | 3165.09 | 0.4533 | 342.59 | 7.24 |

> [!TIP]
> **Nhận xét:** Mô hình **HistGradientBoosting** của Scikit-learn đạt hệ số R² cao nhất (**0.4850**). Tuy nhiên, **LightGBM** cho sai số tuyệt đối MAE thấp nhất (**1748.12**) cùng thời gian huấn luyện nhanh vượt trội (chỉ 347ms so với 932ms của HistGradientBoosting).

---

## 🚀 Hướng dẫn thực thi
Bạn có thể chạy thử nghiệm hoặc đối sánh trực tiếp các mô hình bằng cách mở notebook [04_Model_Benchmark.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/complete_solution/04_Model_Benchmark.ipynb) hoặc chạy thực nghiệm so sánh binning bằng lệnh:
```bash
python3 "practice_2/complete_solution/compare_bins_experiment.py"
```
