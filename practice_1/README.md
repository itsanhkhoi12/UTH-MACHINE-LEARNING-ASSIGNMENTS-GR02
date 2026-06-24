# Practice 1 - Phân loại mail HAM/SPAM

## Define the Problems

### Problems

Trong môi trường doanh nghiệp hiện đại, thư điện tử (Email) là một trong những phương thức trao đổi thông tin quan trọng giữa nhân viên, khách hàng và đối tác. Tuy nhiên, cùng với sự phát triển của Email, số lượng thư rác (SPAM) cũng ngày càng gia tăng, gây ảnh hưởng đến hiệu suất làm việc và tiềm ẩn nhiều rủi ro về bảo mật thông tin.

Một doanh nghiệp đặt ra yêu cầu xây dựng hệ thống có khả năng tự động phân loại các thư điện tử nhận được thành hai nhóm:

* **HAM**: Các email hợp lệ, liên quan đến công việc và cần được xử lý.
* **SPAM**: Các email không mong muốn, quảng cáo hoặc không liên quan đến hoạt động của doanh nghiệp.

Mục tiêu của bài toán là áp dụng các thuật toán `Supervised Learning` nhằm xây dựng mô hình có khả năng tự động phân loại email thành HAM hoặc SPAM dựa trên nội dung thư điện tử.

### Data Context

Bộ dữ liệu được sử dụng trong bài thực hành có nguồn gốc từ tập emails nội bộ của tập đoàn Enron - một trong những tập đoàn năng lượng lớn tại Hoa Kỳ.

Sau vụ bê bối tài chính của Enron, một lượng lớn email nội bộ của công ty đã được công khai và trở thành một trong những bộ dữ liệu phổ biến trong lĩnh vực Data Mining, Machine Learning và Xử lý Ngôn ngữ Tự nhiên Natural Language Processing.


## Workflow

### Load dataset/EDA

* **Data Preview**:
  * Link dataset: Dataset được thu thập từ [GitHub](https://github.com/MWiechmann/enron_spam_data)
  * Số lượng samples ban đầu: 33,716 samples
  * Số lượng samples `SPAM/HAM` trong tập dữ liệu: `17,171/16,545`
  * Các Features chính:
    * `message_id`: Mã định danh của email.
    * `date`: Thời gian gửi email.
    * `subject`: Tiêu đề email.
    * `body`: Nội dung email.
    * `label`: Nhãn phân loại HAM/SPAM.

### Data Preparation

* **Data Preprocessing**:
  * Xử lý dữ liệu khuyết thiếu (Missing Values).
  * Loại bỏ các email bị trùng lặp.
  * Chuyển toàn bộ văn bản về chữ thường.
  * Loại bỏ dấu câu và các ký tự đặc biệt.
  * Masking URL và các số (URL -> `__url__`, số -> `__number__`, dấu chấm than -> `__exclamation__`, ký hiệu tiền tệ -> `__dollar__` để giữ đặc trưng).
  * Loại bỏ các Stopwords.
  * Làm sạch và chuẩn hóa nội dung email.
  * Kết hợp hai cột `Subject` và `Body` thành một thuộc tính văn bản duy nhất là `Text`.

### Feature Engineering

* **Tạo mới đặc trưng**: Xây dựng thuộc tính mới `Punctuation_Ratio` (Tỉ lệ `Punctuation Words` xuất hiện trong `body` content).
* Chia dữ liệu train/test theo tỉ lệ 8/2. 
* **Trích xuất đặc trưng**:
  * Áp dụng thuật toán TF-IDF (Term Frequency - Inverse Document Frequency) tự xây dựng (`TFIDFVectorizerFromScratch`) để chuyển đổi thuộc tính `Text` thành ma trận trọng số từ vựng dưới dạng số.
* **Biến đổi đặc trưng**:
  * Áp dụng phép biến đổi Logarithm cho thuộc tính `Punctuation_Ratio` nhằm giảm hiện tượng lệch phải (Right-Skewed Distribution) do một số email chứa mật độ dấu câu quá lớn.
  * Áp dụng Min-Max Scaling cho thuộc tính đã được Log Transform để đưa dữ liệu về khoảng giá trị `[0, 1]`.
  * Chỉ thực hiện `fit` trên tập huấn luyện và sử dụng cùng tham số đó để `transform` cho tập kiểm thử nhằm tránh hiện tượng rò rỉ dữ liệu (Data Leakage).

### Implement Model From Scratch

Các thuật toán được triển khai hoàn toàn từ đầu nhằm mục tiêu hiểu rõ nguyên lý hoạt động của từng mô hình:
* **Logistic Regression** (Hồi quy Logistic)
* **Random Forest** (Rừng ngẫu nhiên)
* **Support Vector Machine** (SVM phân loại tuyến tính)
* **Naive Bayes** (Multinomial Naive Bayes với Laplace Smoothing và tính toán trong không gian Log)

Mỗi mô hình được huấn luyện trên cùng một tập dữ liệu và cùng quy trình tiền xử lý nhằm đảm bảo tính công bằng trong quá trình so sánh.

### Hyperparameter Tuning

Tiến hành tối ưu siêu tham số cho từng mô hình nhằm cải thiện hiệu suất phân loại.
Các siêu tham số được khảo sát bao gồm:
* **Logistic Regression**:
  * Learning Rate
  * Epochs
* **Random Forest**:
  * Number of Trees
  * Maximum Depth
  * Minimum Samples Split
* **SVM**:
  * Learning Rate
  * Lambda Parameter
* **Naive Bayes**:
  * Alpha (Laplace Smoothing factor)

---

## 📊 Chỉ Số Đánh Giá & Bảng So Sánh Hiệu Năng

Vì bài toán phân loại email spam thường gặp hiện tượng mất cân bằng dữ liệu, việc sử dụng chỉ số **Accuracy** là chưa đủ. Dự án tập trung tối ưu hóa các chỉ số sau:
* **Precision**: Tỷ lệ thư được dự đoán là Spam thực sự là Spam (cần cực kỳ cao để tránh lọc nhầm email công việc Ham).
* **Recall**: Tỷ lệ thư Spam thực tế được mô hình phát hiện ra.
* **F1-Score**: Điểm trung bình điều hòa giữa Precision và Recall.

### Bảng Kết quả Chạy Thực tế (Test Set)

| Mô hình | Thuật toán | Loại | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | Multinomial NB | Scikit-learn | **98.29%** | **98.48%** | **98.16%** | **98.32%** |
| | Multinomial NB | From Scratch | **98.29%** | **98.48%** | **98.16%** | **98.32%** |
| **SVM** | Linear SVM | Scikit-learn | 98.36% | 97.86% | 98.69% | 98.27% |
| | Linear SVM | From Scratch | 88.48% | 98.61% | 76.74% | 86.31% |
| **Logistic Regression** | Logistic Regression | From Scratch | 83.73% | 90.79% | 73.03% | 80.95% |
| | Logistic Regression | Scikit-learn | 83.17% | 98.38% | 65.51% | 78.65% |
| **Random Forest** | Random Forest | From Scratch | 77.07% | 97.18% | 53.09% | 68.66% |

> [!NOTE]
> Kết quả khớp tuyệt đối đến chữ số thập phân thứ 6 giữa bản **Naive Bayes Scratch** và Scikit-learn khẳng định thuật toán tự viết chính xác 100% về mặt toán học.

---

## 🏆 Dashboard Tương Tác Chọn Ngưỡng (Interactive Dashboard)

Dự án cung cấp trang Dashboard offline **`threshold_selector.html`** cho phép người dùng kéo thanh trượt thay đổi ngưỡng xác suất phân loại (Threshold từ 0.00 đến 1.00) để trực quan hóa sự thay đổi theo thời gian thực của Precision, Recall, F1, FPR, Confusion Matrix và quan sát các email mẫu ranh giới bị dự đoán sai.

---

## 📦 Lưu Trữ Mô Hình (Models Serialization)

Các mô hình được huấn luyện tốt nhất phục vụ cho việc dự đoán sau này được đóng gói tại thư mục `practice_1/models/`:
* `Linear_SVM_Scratch.pkl`
* `Logistic_Regression.pkl`
* `Naive_Bayes_Scratch.pkl`
* `Random_Forest_Scratch.pkl`

---

## Hướng phát triển

* Mở rộng từ bài toán phân loại nhị phân (HAM/SPAM) sang bài toán phân loại nhiều nhóm Email khác nhau.
