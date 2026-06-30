# PRACTICE 2: PREDICTING PRODUCT SALES
## Workflow of Machine Learning Pipeline

### **1. Define Problem**
- Một tập đoàn bán lẻ quản lý chuỗi cửa hàng phân phối tại 10 trung tâm thương mại sầm uất nhất Istanbul (Thổ Nhĩ Kỳ) đang cần giải quyết bài toán quản lý luân chuyển hàng hóa và dòng tiền. Họ muốn dự đoán Doanh thu của từng giao dịch dựa trên thông tin nhân khẩu học của khách hàng và bối cảnh mua sắm.Việc dự đoán được mức chi tiêu này sẽ giúp phía doanh nghiệp đạt được 2 mục tiêu chiến lược:
    - **Tối ưu hóa hàng tồn kho:** Phân bổ chính xác số lượng và loại hàng hóa (Ví dụ: Công nghệ, Quần áo, Thực phẩm) cho từng trung tâm thương mại tại từng thời điểm cụ thể, nhằm giảm chi phí lưu kho và tránh tình trạng "cháy hàng"
    - **Chiến lược Marketing:** Thiết kế các chương trình khuyến mãi cá nhân hóa nhắm đúng vào tệp khách hàng tiềm năng (dựa trên độ tuổi, giới tính, phương thức thanh toán) để tối đa hóa lợi nhuận

### **2. Collect Data**
Bộ dữ liệu bao gồm `99,457` mẫu giao dịch (hóa đơn) với 10 cột thông tin chi tiết (Data Source: [Kaggle](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset))

**Các biến đặc trưng (Feaures - X):**

| Tên cột | Mô tả chi tiết |
|---------|----------------|
|invoice_no|Mã hóa đơn (Chuỗi định danh duy nhất cho mỗi giao dịch)|
|customer_id|Mã định danh khách hàng|
|gender|Giới tính của khách hàng (Male / Female)|
|age|Độ tuổi của khách hàng|
|category|Ngành hàng của sản phẩm được mua (Clothing, Cosmetics, Technology...)|
|quantity|Số lượng sản phẩm được mua trong đơn hàng|
|payment_method|Phương thức thanh toán (Cash, Credit Card, Debit Card)|
|invoice_date|Ngày thực hiện giao dịch mua sắm|
|shopping_mall|Tên chi nhánh trung tâm thương mại (10 trung tâm lớn tại Istanbul)|

**Biến mục tiêu (Target Variable - y):**

| Tên cột | Mô tả chi tiết |
|---------|----------------|
|price|Giá trị của đơn hàng (Đại diện cho Doanh thu/Revenue). Đây là biến số liên tục|

### **3. Exploratory Data Analysis (EDA)**
**Mục tiêu :** Khám phá cấu trúc bộ dữ liệu, phát hiện các quy luật, xu hướng ẩn, và nhận diện những điểm bất thường. Đánh giá mối tương quan giữa các biến độc lập và biến mục tiêu (Doanh thu) để tạo tiền đề logic và thống kê vững chắc cho các bước tiền xử lý và chọn lọc đặc trưng sau này.
#### **Data Overview**
- Kiểm tra tổng số lượng mẫu (samples) và số lượng đặc trưng (features)
- Kiểm tra kiểu dữ liệu (data type) của từng cột
- Thống kê các đại lượng mô tả cơ bản (Mean, Min, Max, Std...) cho các biến số
- Query một vài dòng mẫu để nắm bắt trực quan cấu trúc của dữ liệu

#### **Missing Data Analysis**
- Tính toán tỷ lệ phần trăm dữ liệu khuyết thiếu (missing values) trên từng đặc trưng riêng biệt
- Trực quan hóa bằng biểu đồ Seaborn Heatmap để định vị cụ thể vị trí và đánh giá mức độ phân bổ của dữ liệu bị thiếu

#### **Univariate Analysis**
- **Nhóm Numerical (Tuổi, Số lượng, Giá trị đơn hàng - Đại diện cho Doanh thu):** Trực quan hóa bằng Distribution Plot để đánh giá hình dáng phân phối và độ lệch
- **Nhóm Categorical (Giới tính, Ngành hàng, Thanh toán):** Trực quan hóa bằng Barplot/Countplot để kiểm tra tần suất và độ cân bằng của các lớp (class)
- **Nhóm Time Series:** Trích xuất ra thành phần Tháng và Ngày trong tuần để trực quan hóa đồ thị tần suất đánh giá lưu lượng khách hàng tập trung vào thời điểm nào trong năm/tuần

#### **Bivariate/Multivariate Analysis**
- **Numerical vs Numerical:** Sử dụng Ma trận tương quan giữa các biến số gồm age, quantity, price để kiểm tra hiện tượng đa cộng tuyến (Ví dụ: age và quantity có tương quan với nhau không ?, ...). Vẽ Scatter Plots để xem xét xu hướng tuyến tính giữa age vs price, age vs quantity
- **Categorical vs Numerical:** Sử dụng Boxplot và Barplot để tìm ra ngành hàng mang lại trung vị giá trị cao nhất, Mall có phổ doanh thu rộng nhất, và so sánh doanh thu theo giới tính/phương thức thanh toán
- **Categorical vs Categorical:** Sử dụng Bảng chéo (pd.crosstab) và Stacked Bar Charts để tìm Insight nhân khẩu học (phân bổ giới tính tại các Mall) và Insight hành vi (sở thích ngành hàng/thanh toán theo giới tính)
- **Time Series vs Numerical:** Sử dụng Barplot/Linechart để xem xét giá trị hóa đơn trung bình thay đổi thế nào theo Tháng (Tháng 12 có tăng vọt không?) và theo Ngày trong tuần (Cuối tuần có chi tiêu cao hơn không?)

#### **Outlier Detection**
- Tập trung phân tích phân phối đuôi dài của biến price (Tổng hóa đơn - đại diện cho Doanh thu)
- Sử dụng phương pháp IQR để đếm số lượng hóa đơn có giá trị "bất thường" (mua quá nhiều hoặc giá quá cao)

### **4. Data Preprocessing**
**Mục tiêu :** Xử lý triệt để các vấn đề của dữ liệu thô (nhiễu, lỗi định dạng, thiếu sót, trùng lặp) nhằm tạo ra một bộ dữ liệu sạch, đồng nhất và chuẩn xác. Đảm bảo mô hình không bị "nhiễu" bởi các dữ liệu phi logic hoặc sai sót trong quá trình thu thập.

Các bước thực hiện:
- **Data Cleaning:**
    - Xử lý Duplicate: Quét và xóa bỏ các dòng giao dịch trùng lặp hoàn toàn.
    - Xử lý Inconsistent Type: Ép kiểu cột invoice_date về định dạng Datetime, cột price về Float.
    - Xử lý Invalid/Noisy Data: Chuẩn hóa lỗi gõ chữ (Ví dụ: Đưa "Male", "male", "M" về chuẩn chung là "Male").
    - Xử lý Domain Constraints: Xóa các dòng phi logic (VD: age < 10, price < 0).
    - Xử lý Missing Values: Xóa (Drop) các dòng chứa giá trị NaN nếu tỷ lệ rất thấp
    - Xử lý Outlier: Dùng phương pháp IQR để phát hiện và Cắt tỉa/Xóa bỏ các hóa đơn có giá trị dị biệt, giúp làm sạch dữ liệu trước khi đưa vào mô hình

### **5. Feature Engineering**
**Mục tiêu :**
- Tạo ra các biến mới để mô hình bắt được các quy luật mua sắm phức tạp
- Loại bỏ các cột không mang thông tin để giảm nhiễu
- Giảm chiều dữ liệu và chuẩn hóa thang đo giúp thuật toán hội tụ nhanh hơn
- Giúp mô hình hoạt động tốt trên tập Test và dữ liệu thực tế

Các bước thực hiện:
- **Data split:** Phân chia tập dữ liệu thành Train/Test theo tỷ lệ 80/20 trước khi tác động biến đổi
- **Feature Extraction:** Trích xuất biến thời gian thành biến cờ Is_Promotion_Campaign (1 nếu thuộc Tháng 1 hoặc Tháng 2, ngược lại 0)
- **Feature Transformation:**
    - **Log-Transform:** Biến mục tiêu price (Doanh thu) bị lệch phải nghiêm trọng cần biến đối logarit để kéo phân phối về hình chuông, giúp thỏa mãn giả định của Linear Regression
    - **Rời rạc hóa (Binning):** Chuyển đổi cột age (tuổi từ 18-90) thành các nhóm (Bins) như: Gen Z (<25), Millennials (25-40), Gen X (41-60), Boomers (>60). Việc này giúp mô hình dễ dàng học được sự khác biệt về chi tiêu giữa các thế hệ thay vì coi tuổi tác là một đường thẳng tuyến tính
- **Feature Selection:** Loại bỏ các cột không mang thông tin dự báo (invoice_no, customer_id, invoice_date) và các biến phân loại gây nhiễu (shopping_mall, gender, payment_method)
- **Encoding Categorical Variables:** Đưa các biến phân loại (category, age_group) qua bộ One-Hot Encoding (OHE) để chuyển đổi thành ma trận nhị phân
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