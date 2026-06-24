# Practice 2 - Dự đoán Doanh số Sản phẩm (Product Sales Regression)

## Define the Problems

### Problems

Trong quản lý bán lẻ và chuỗi cung ứng, việc dự đoán chính xác doanh số bán hàng (Sales Revenue) đóng vai trò quyết định giúp tối ưu hóa lượng hàng lưu kho, lập kế hoạch khuyến mãi và giảm thiểu chi phí vận hành. 

Bài toán đặt ra là xây dựng hệ thống hồi quy học máy (Regression) để dự đoán doanh số bán hàng dựa trên thuộc tính giao dịch. Dự án được chia làm hai cấp độ dự báo:
1. **Dự báo cấp độ giao dịch đơn lẻ (Invoice Level)**: Dự đoán trực tiếp doanh thu của từng hóa đơn.
2. **Dự báo cấp độ chuỗi thời gian gộp tuần (Weekly Aggregated Level)**: Dự đoán doanh thu tuần gộp của từng ngành hàng tại các trung tâm thương mại khác nhau, giúp giảm thiểu biến động ngẫu nhiên.

Mục tiêu là áp dụng các mô hình `Supervised Learning` (hồi quy tuyến tính, cây quyết định và các thuật toán học máy nhóm Boosting) để dự báo chính xác doanh số bán hàng.

### Data Context

Bộ dữ liệu sử dụng là thông tin giao dịch bán lẻ lịch sử (Retail Transactions) chứa chi tiết về mã hóa đơn, thông tin khách hàng (tuổi, giới tính), sản phẩm (ngành hàng, giá bán, số lượng), địa điểm giao dịch (trung tâm thương mại) và ngày mua hàng.

Sau quá trình xử lý làm sạch, dữ liệu được gom nhóm theo **Tuần, Ngành hàng và Trung tâm thương mại** để tạo ra tập dữ liệu chuỗi thời gian ổn định phục vụ dự báo xu hướng dài hạn.


## Workflow

### Load dataset/EDA

* **Khảo sát dữ liệu**:
  * Phát hiện và sửa đổi logic kiểm tra tuổi của khách hàng (giới hạn từ 10 đến 100 tuổi).
  * Chuyển đổi thuộc tính ngày tháng sang định dạng thời gian tiêu chuẩn.
* **Phân tích khám phá & Chuỗi thời gian**:
  * Phân tích hàm tự tương quan (Autocorrelation) và phát hiện hệ số tự tương quan **Lag 1 đạt tới 0.78**, cho thấy tính phụ thuộc thời gian mạnh mẽ.
  * Thực hiện kiểm định ANOVA chứng minh sự phụ thuộc hoàn toàn của doanh thu vào ngành hàng và trung tâm thương mại (p-value = 0), trong khi các thuộc tính nhân khẩu học (tuổi, giới tính) phân bố đồng đều (p-value > 0.70) ở mức gộp tuần.

### Data Preparation

* **Gom nhóm theo Tuần (Weekly Aggregation)**:
  * Dữ liệu giao dịch được gộp theo `Weekly`, `Category`, và `Shopping Mall`.
  * Tính tổng doanh thu tuần (`Sales_Revenue`), tổng số lượng bán (`Quantity`), và số lượng giao dịch (`Trans_Count`).
  * Tạo ra tập dữ liệu chuỗi thời gian gồm 8,360 dòng dữ liệu sạch.

### Feature Engineering

* **Phân chia tập dữ liệu**:
  * Chia dữ liệu Train/Test theo trục thời gian (trước ngày 01/11/2022 làm tập Train, sau ngày 01/11/2022 làm tập Test) để tránh hiện tượng rò rỉ dữ liệu tương lai.
* **Trích xuất đặc trưng chuỗi thời gian (Temporal Features)**:
  * Tạo các đặc trưng trễ: `Lag_1`, `Lag_2`, và `Lag_4` để đưa thông tin doanh số quá khứ vào mô hình.
  * Tạo các đặc trưng thống kê trượt: `rolling_mean_4` và `rolling_std_4` (trung bình và độ lệch chuẩn trượt 4 tuần) để bắt xu thế ngắn hạn.
* **Biến đổi chu kỳ thời gian (Cyclic Encoding)**:
  * Mã hóa Sin/Cos cho thuộc tính tuần học và tháng học để giữ nguyên tính chu kỳ tuần hoàn liên tục của thời gian (tuần 52 sát tuần 1).

### Implement Model From Scratch

Các thuật toán được huấn luyện trên cùng một tập dữ liệu đặc trưng chuỗi thời gian:
* **Hồi quy tuyến tính** (Linear Regression)
* **Cây quyết định hồi quy phát triển theo lá** (Leaf-Wise Decision Tree Regressor)
* **LightGBM tự cài đặt từ đầu** (LightGBM From Scratch):
  * Cài đặt phân nhóm Histogram (Histogram-based Feature Binning) với 32 bins để tối ưu hóa tìm kiếm điểm chia nhánh.
  * Lập trình cấu trúc cây Leaf-wise regressor chọn lá tốt nhất dựa trên Gain có phạt Regularization L2 để chống overfitting.

### Hyperparameter Tuning

Tiến hành tối ưu siêu tham số cho từng mô hình Boosting thông qua GridSearchCV và K-Fold:
* **Linear Regression**:
  * Learning Rate
  * Epochs
* **Decision Tree**:
  * Maximum Depth
  * Maximum Leaf Nodes
  * Minimum Samples Split
* **LightGBM From Scratch / Thư viện**:
  * Number of Trees (n_estimators)
  * Learning Rate
  * Regularization L2 (l2_reg)
  * Maximum Leaf Nodes
  * Number of Bins (n_bins)

---

## 📊 Chỉ Số Đánh Giá & Bảng So Sánh Hiệu Năng

Đối với bài toán hồi quy, mô hình được đánh giá qua 3 chỉ số chính:
* **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình, đo lường độ lệch doanh số tuần.
* **RMSE (Root Mean Squared Error)**: Đo sai số bình phương trung bình, giúp phát hiện độ nhạy đối với các biến động nhu cầu cực đoan (outliers).
* **R² Score (R-squared)**: Hệ số xác định khả năng giải thích phương sai của mô hình.

### 1. Kết quả trên dữ liệu Hóa đơn đơn lẻ (Invoice Level)

| Mô hình | Thuật toán | Loại | MAE | RMSE | R² Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | Histogram Gradient Boosting | Scikit-learn | 1751.37 | 3072.06 | **0.4850** |
| **LightGBM** | LGBMRegressor | Thư viện ngoài | **1748.12** | 3078.36 | 0.4829 |
| **Decision Tree** | DecisionTreeRegressor | From Scratch | - | **3033.82** | 0.4821 |
| **Linear Regression** | Ordinary Least Squares | Scikit-learn | 1752.12 | 3055.46 | 0.4815 |
| **CatBoost** | CatBoostRegressor | Thư viện ngoài | 1772.32 | 3117.80 | 0.4695 |
| **XGBoost** | XGBRegressor | Thư viện ngoài | 1787.86 | 3165.09 | 0.4533 |
| **Linear Regression** | Gradient Descent | From Scratch | 2450.45 | 3409.55 | 0.3850 |

### 2. Kết quả trên dữ liệu Chuỗi thời gian Gộp tuần (Weekly Aggregated Level)

| Hạng | Mô hình | Loại | MAE | RMSE | R² Score | Thời gian chạy (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **HistGradientBoosting** | Scikit-learn | **1873.22** | **4334.42** | **0.8798** | 0.73 |
| 2 | **XGBoost** | Thư viện ngoài | 1998.28 | 4418.57 | 0.8751 | 0.24 |
| 3 | **LightGBM** | Thư viện ngoài | 1914.08 | 4431.19 | 0.8744 | 0.29 |
| 4 | **CatBoost** | Thư viện ngoài | 2065.46 | 4456.27 | 0.8729 | 0.21 |
| 5 | **LightGBM (Scratch)** | From Scratch | 2035.06 | 4504.85 | 0.8702 | 2.04 |

> [!IMPORTANT]
> **Nhận xét chuyên môn:** Việc gộp tuần (Weekly Aggregation) giúp tăng đáng kể hệ số xác định $R^2$ từ **~48.5% lên ~88.0%** nhờ triệt tiêu nhiễu ngẫu nhiên theo định luật số lớn. Tuy nhiên, sai số tuyệt đối trung bình **MAE vẫn chiếm 26.6%** so với trung bình tuần, và **RMSE cao gấp đôi MAE** cho thấy mô hình còn nhạy cảm với các đỉnh doanh thu đột biến (peaks) hoặc mùa khuyến mãi.

---

## Hướng phát triển

* Áp dụng phép biến đổi Logarit $y_{\text{new}} = \log(y + 1)$ cho biến doanh thu lệch phải để giảm thiểu RMSE.
* Bổ sung các thông tin ngoại cảnh như chương trình khuyến mãi (promotions), sự kiện ngày lễ (Black Friday, Tết), và các chỉ số kinh tế vĩ mô.
* Triển khai công thức tính toán lượng tồn kho an toàn (Safety Stock) tích hợp trực tiếp từ sai số RMSE của mô hình dự báo.
