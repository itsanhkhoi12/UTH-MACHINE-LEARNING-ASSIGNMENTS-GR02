# Bài Thực Hành 2: Dự Đoán Doanh Số Sản Phẩm (Predicting Product Sales)

Bài thực hành này tập trung vào bài toán **Hồi quy (Regression)** để dự đoán doanh số bán hàng (Sales Revenue) dựa trên các thuộc tính của sản phẩm và cửa hàng. Dự án thử nghiệm nhiều lớp mô hình từ các thuật toán tuyến tính đơn giản đến các mô hình học cây phức tạp (Decision Tree) và các thuật toán nâng cao nhóm Boosting (XGBoost, LightGBM, CatBoost, HistGradientBoosting).

---

## 📂 Danh Sách Các Bước Thực Hiện (Notebooks)

Tiến trình thực hiện được chia nhỏ thành các file Notebook theo quy chuẩn:
1. [01-02-eda-data-preprocessing.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/01-02-eda-data-preprocessing.ipynb): Phân tích phân phối của doanh số (biến mục tiêu), xử lý giá trị thiếu (missing values), chuẩn hóa dữ liệu và mã hóa biến phân loại.
2. [03-feature-engineering.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/03-feature-engineering.ipynb): Tạo thêm đặc trưng mới, trích xuất thông tin, xử lý tương quan đa cộng tuyến và lựa chọn đặc trưng.
3. [04-train-linear-regression.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/04-train-linear-regression.ipynb): Xây dựng mô hình Hồi quy tuyến tính Linear Regression (Sklearn vs. Scratch) kết hợp chính quy hóa Ridge/Lasso.
4. [05-train-decision-tree-regressor.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/05-train-decision-tree-regressor.ipynb): Xây dựng mô hình Cây quyết định hồi quy (Decision Tree Regressor) từ đầu.
5. [06-train-gradient-boosting.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2/06-train-gradient-boosting.ipynb): Huấn luyện và tinh chỉnh các siêu tham số cho các mô hình Gradient Boosting mạnh mẽ.

---

## 📊 Chỉ Số Đánh Giá (Metrics)

Đối với bài toán hồi quy, mô hình được đánh giá qua 3 chỉ số chính:
- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình, đo lường độ lệch trung bình giữa doanh số dự đoán và doanh số thực tế.
- **RMSE (Root Mean Squared Error)**: Căn sai số bình phương trung bình. Chỉ số này phạt nặng hơn các sai số lớn (outliers).
- **R² Score (R-squared)**: Hệ số xác định, cho biết tỷ lệ phần trăm biến thiên của doanh số mục tiêu được giải thích bởi mô hình (càng gần 1.0 càng tốt).

---

## 🏆 Bảng So Sánh Hiệu Năng Các Mô Hiện Tại (Benchmark)

Dưới đây là kết quả đối sánh hiệu năng các mô hình được ghi nhận trên tập dữ liệu kiểm thử (Test Set):

| Mô hình | Thuật toán | Loại | MAE | RMSE | R² Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | Histogram Gradient Boosting | Scikit-learn | 1751.37 | 3072.06 | **0.4850** |
| **LightGBM** | LGBMRegressor | Thư viện ngoài | **1748.12** | 3078.36 | 0.4829 |
| **Decision Tree** | DecisionTreeRegressor | From Scratch | - | **3033.82** | 0.4821 |
| **Linear Regression** | Ordinary Least Squares | Scikit-learn | 1752.12 | 3055.46 | 0.4815 |
| **CatBoost** | CatBoostRegressor | Thư viện ngoài | 1772.32 | 3117.80 | 0.4695 |
| **XGBoost** | XGBRegressor | Thư viện ngoài | 1787.86 | 3165.09 | 0.4533 |
| **Linear Regression** | Gradient Descent | From Scratch | 2450.45 | 3409.55 | 0.3850 |

> [!TIP]
> **Nhận xét kết quả:** Mô hình **HistGradientBoosting (Sklearn)** và **LightGBM** đang dẫn đầu về độ chính xác và khả năng giải thích dữ liệu với R² Score xấp xỉ **0.485**. Đồng thời, thuật toán LightGBM có thời gian huấn luyện cực kỳ tối ưu (chỉ khoảng 347ms).

---

## 📌 Hướng Đi Tiếp Theo (Next Steps)
- Tận dụng quy trình **Preprocessing Alignment** để làm sạch và xử lý phân phối lệch của doanh số, đặc biệt áp dụng các hàm transform (như Log Transform) trước khi đưa vào huấn luyện mô hình đã chọn.
- Xây dựng thêm thư mục lưu trữ mô hình `models/` riêng cho Bài thực hành 2 để đóng gói các file `.pkl`.
