# PRACTICE 2: PREDICTING PRODUCT SALES
## Workflow of Machine Learning Pipeline

### **1. Define Problem**

### **2. Collect Data**
Data Source: [Kaggle](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset/data)

**Các biến đặc trưng (Feaures - X):**

| Tên cột | Mô tả chi tiết |
|---------|----------------|
|TransactionID|Mỗi giao dịch được định danh duy nhất, cho phép theo dõi và tham chiếu|
|Date|Ngày diễn ra giao dịch, cung cấp thông tin chi tiết về xu hướng bán hàng theo thời gian|
|Customer ID|Mỗi khách hàng được định danh duy nhất, cho phép phân tích tập trung vào khách hàng|
|Gender|Giới tính của khách hàng (Nam/Nữ), cung cấp thông tin chi tiết về mô hình mua hàng dựa trên giới tính|
|Age|Độ tuổi của khách hàng, tạo điều kiện thuận lợi cho việc phân khúc và khám phá các yếu tố ảnh hưởng liên quan đến độ tuổi|
|Product Category|Loại sản phẩm đã mua (ví dụ: Đồ điện tử, Quần áo, Mỹ phẩm), giúp hiểu rõ hơn về sở thích sản phẩm|
|Quantity|Số lượng đơn vị sản phẩm được mua, góp phần cung cấp thông tin chi tiết về khối lượng mua hàng|
|Price per Unit|Giá của một đơn vị sản phẩm, hỗ trợ trong các tính toán liên quan đến tổng chi tiêu|

**Biến mục tiêu (Target Variable - y):**

| Tên cột | Mô tả chi tiết |
|---------|----------------|
|Total Amount|Tổng giá trị tiền tệ của giao dịch đó (Đại diện cho Doanh thu/Revenue). Đây là biến số liên tục|

### **3. Exploratory Data Analysis (EDA)**
**Mục tiêu :** Khám phá cấu trúc bộ dữ liệu, phát hiện các quy luật, xu hướng ẩn, và nhận diện những điểm bất thường. Đánh giá mối tương quan giữa các biến độc lập và biến mục tiêu để tạo tiền đề logic và thống kê vững chắc cho các bước tiền xử lý và chọn lọc đặc trưng sau này.
#### **Data Overview**
- Kiểm tra tổng số lượng mẫu (samples) và số lượng đặc trưng (features)
- Kiểm tra kiểu dữ liệu (data type) của từng cột
- Thống kê các đại lượng mô tả cơ bản (Mean, Min, Max, Std...) cho các biến số
- Query một vài dòng mẫu để nắm bắt trực quan cấu trúc của dữ liệu
#### **Missing Data Analysis**
- Tính toán tỷ lệ phần trăm dữ liệu khuyết thiếu (missing values) trên từng đặc trưng riêng biệt
- Trực quan hóa bằng biểu đồ Seaborn Heatmap để định vị cụ thể vị trí và đánh giá mức độ phân bổ của dữ liệu bị thiếu
#### **Univariate Analysis**
- **Nhóm Numerical (Age, Quantity, Price per Unit, Total Amount):** Trực quan hóa bằng Distribution Plot để đánh giá hình dáng phân phối và độ lệch
- **Nhóm Categorical (Gender, Product Category):** Trực quan hóa bằng Barplot/Countplot để kiểm tra tần suất và độ cân bằng của các lớp
- **Nhóm Time Series:** Trích xuất ra thành phần Tháng và Ngày trong tuần để trực quan hóa đồ thị tần suất đánh giá nhịp điệu mua sắm
#### **Bivariate/Multivariate Analysis**
- **Numerical vs Numerical:** Sử dụng Ma trận tương quan giữa các biến số gồm Age, Quantity, Price per Unit, Total Amount để kiểm tra hiện tượng đa cộng tuyến. Vẽ Scatter Plots để xem xét xu hướng tuyến tính giữa các cặp Quantity vs Total Amount, Price per Unit vs Total Amount, Age vs Total Amount
- **Categorical vs Numerical:** Product Category vs Total Amount, Gender vs Total Amount
- **Categorical vs Categorical:** Gender vs Product Category
- **Time Series vs Numerical:** Month/DayOfWeek vs Total Amount
- **Time Series vs Categorical:** Month vs Product Category

#### **Outlier Detection**
- Tập trung phân tích phân phối của Total Amount
- Sử dụng phương pháp IQR để đếm số lượng hóa đơn có giá trị "bất thường" 

### **4. Data Preprocessing**
**Mục tiêu :** Xử lý triệt để các vấn đề của dữ liệu thô (nhiễu, lỗi định dạng, thiếu sót, trùng lặp) nhằm tạo ra một bộ dữ liệu sạch, đồng nhất và chuẩn xác. Đảm bảo mô hình không bị "nhiễu" bởi các dữ liệu phi logic hoặc sai sót trong quá trình thu thập.
Các bước thực hiện:
    - Data Split: Chia tập Train/Test tỷ lệ 80/20
    - Xử lý Data Leakage: Xóa bỏ các cột ID không mang thông tin quy luật (Transaction ID, Customer ID) trên tập Train và Test để giảm kích thước lưu trữ và tránh Data Leakage
    - Xử lý Duplicate: Quét và xóa bỏ các bản ghi trùng lặp hoàn toàn trên tập Train
    - Xử lý Inconsistent Type: Ép kiểu toàn bộ các cột về định dạng chuẩn (vd: Date $\rightarrow$ datetime64, Gender $\rightarrow$ category, các biến số $\rightarrow$ float) trên cả Train và Test
    - Xử lý Invalid/Noisy Data: Xóa các dòng chứa giá trị phi logic (VD: age < 10, price < 0) trên tập Train
    - Xử lý Domain Constraints: Kiểm tra các ràng buộc thực tế (vd: Nếu Product Category chỉ được phép là [Beauty, Clothing, Electronics], thì bất kỳ giá trị khác phải bị loại bỏ hoặc đưa về nhóm Unknown/Other) trên tập Train

### **5. Feature Engineering**
**Mục tiêu :**
- Tạo ra các biến mới để mô hình bắt được các quy luật mua sắm phức tạp
- Loại bỏ các cột không mang thông tin để giảm nhiễu
- Giảm chiều dữ liệu và chuẩn hóa thang đo giúp thuật toán hội tụ nhanh hơn
- Giúp mô hình hoạt động tốt trên tập Test và dữ liệu thực tế

Các bước thực hiện:
- **Feature Creation:** Thực hiện trên cả Train lẫn Test
    - **Date Features:** Trích xuất các thuộc tính thời gian: `Month`, `DayOfWeek` (từ `Date`)
    - **Domain-specific Features:** Tạo biến `Is_Weekend` (Binary: 1 nếu Sat/Sun, 0 ngược lại) để tận dụng kết quả EDA về sự chênh lệch chi tiêu cuối tuần
    - **Interaction Features:** Nhân tố liên quan giữa các biến: Tạo biến `Gender_x_Category` để mô hình học được hành vi đặc thù (ví dụ: Nữ + Beauty)
- **Feature Selection:** Loại bỏ các cột không mang thông tin dự báo (Date, Gender, Quantity, Product Category gốc) nhưng giữ lại cột Price per Unit
- **Encoding Categorical Variables:** Đưa các biến dạng Category qua bộ One-Hot Encoding (OHE) để chuyển đổi thành ma trận nhị phân (Chỉ thực hiện .fit_transform() trên tập Train và áp dụng .transform() trên tập Test)
- **Feature Transformation:**
    - **Log-Transform:** Biến mục tiêu `Total Amount` bị lệch phải cần biến đối logarit để kéo phân phối về hình chuông
    - **Scaling (Chuẩn hóa thang đo):** Chuẩn hóa Standardization (Z-score) cho các biến số liên tục để kéo Mean = 0 và Std = 1. Việc này giúp thuật toán Gradient Descent trong Linear Regression không bị chệch hướng bởi các biến có giá trị quá lớn

### **6. Model Training**
Triển khai huấn luyện các thuật toán Regression, kết hợp với các kỹ thuật kiểm định và tinh chỉnh để đạt được hiệu năng dự đoán tốt nhất:
- **Huấn luyện mô hình cơ sở:**
    - Linear Regression
    - Decision Tree Regression
    - Random Forest Regression
    - Gradient Boosting Regression
- **Cross Validation:** Áp dụng kỹ thuật K-Fold Cross Validation (ví dụ: K=5) trên tập Huấn luyện để kiểm tra tính ổn định của mô hình, đảm bảo kết quả không bị phụ thuộc vào cách chia dữ liệu và giảm thiểu rủi ro Overfitting
- **Hyperparameter Optimization - HPO:** Sử dụng kỹ thuật GridSearchCV để dò tìm tự động không gian siêu tham số, từ đó tìm ra cấu hình tốt nhất (best parameters) cho các thuật toán phức tạp như Random Forest hay Gradient Boosting

### **7. Model Evaluation**
Đánh giá chất lượng mô hình thông qua các bộ chỉ số đo lường sai số trên tập Test:
- **R² (R-squared):** Hệ số xác định mức độ giải thích của mô hình đối với sự biến thiên của dữ liệu.
- **RMSE (Root Mean Squared Error):** Căn bậc hai sai số toàn phương trung bình
- **MAE (Mean Absolute Error):** Sai số tuyệt đối trung bình
- **MAPE (Mean Absolute Percentage Error):** Phần trăm sai số tuyệt đối trung bình