# Practice 1 - Phân loại mail HAM/SPAM

## Define the Problems

### Problems

Trong môi trường doanh nghiệp hiện đại, thư điện tử (Email) là một trong những phương thức trao đổi thông tin quan trọng giữa nhân viên, khách hàng và đối tác. Tuy nhiên, cùng với sự phát triển của Email, số lượng thư rác (SPAM) cũng ngày càng gia tăng, gây ảnh hưởng đến hiệu suất làm việc và tiềm ẩn nhiều rủi ro về bảo mật thông tin.

Một doanh nghiệp đặt ra yêu cầu xây dựng hệ thống có khả năng tự động phân loại các thư điện tử nhận được thành hai nhóm:

* HAM: Các email hợp lệ, liên quan đến công việc và cần được xử lý.
* SPAM: Các email không mong muốn, quảng cáo hoặc không liên quan đến hoạt động của doanh nghiệp.

Mục tiêu của bài toán là áp dụng các thuật toán `Supervised Learning` nhằm xây dựng mô hình có khả năng tự động phân loại email thành HAM hoặc SPAM dựa trên nội dung thư điện tử.

### Data Context

Bộ dữ liệu được sử dụng trong bài thực hành có nguồn gốc từ tập emails nội bộ của tập đoàn Enron - một trong những tập đoàn năng lượng lớn tại Hoa Kỳ.

Sau vụ bê bối tài chính của Enron, một lượng lớn email nội bộ của công ty đã được công khai và trở thành một trong những bộ dữ liệu phổ biến trong lĩnh vực Data Mining, Machine Learning và Xử lý Ngôn ngữ Tự nhiên Natural Language Processing.


## Workflow

### Load dataset/EDA

**Mục tiêu:** Hiểu rõ cấu trúc dữ liệu, đánh giá chất lượng, xử lý các giá trị khuyết thiếu và ngoại lai (outliers) nhằm tạo tiền đề vững chắc cho các bước tiền xử lý tiếp theo.

* Data Preview:

  * Link dataset: Dataset được thu thập từ [GitHub](https://github.com/MWiechmann/enron_spam_data)
  * Số lượng samples ban đầu: 33716 samples
  * Số lượng samples `SPAM/HAM` trong tập dữ liệu: `17171/16545`
  * Các Features chính:

    * `message_id`: Mã định danh của email.
    * `date`: Thời gian gửi email.
    * `subject`: Tiêu đề email.
    * `body`: Nội dung email.
    * `label`: Nhãn phân loại HAM/SPAM.

### Data Preparation

**Mục tiêu:** Biến đổi dữ liệu văn bản thô, phi cấu trúc thành văn bản sạch, chuẩn hóa để sẵn sàng cho mô hình học máy.

* **Data Cleaning:**
  * **Xử lý khuyết thiếu & Trùng lặp:** Điền khuyết thiếu từ bước EDA và loại bỏ các email bị trùng lặp nội dung.
  * **Hợp nhất Đặc trưng (Feature Merging):** Ghép cột `Subject` và `Message` thành một đặc trưng duy nhất là `Full_Text`, sau đó loại bỏ 2 cột gốc.
  * **Masking Dữ liệu (Regex):** 
    * Che giấu các liên kết web thành `URLTOKEN`.
    * Che giấu các con số và ký hiệu tiền tệ thành `NUMTOKEN`.
  * **Chuẩn hóa cơ bản:** Chuyển toàn bộ văn bản về chữ thường (Lowercasing); xóa bỏ các thẻ HTML nhúng trong email; loại bỏ dấu câu, ký tự đặc biệt và chuẩn hóa khoảng trắng.

* **Text Transformation:**
  * **Tokenization:** Tách văn bản thành các từ vựng đơn lẻ (tokens).
  * **POS Tagging:** Gắn thẻ từ loại (Danh từ, động từ...) để cung cấp ngữ cảnh cho bước chuẩn hóa từ.
  * **Lemmatization:** Đưa các từ vựng về nguyên bản từ điển (VD: *running* $\rightarrow$ *run*).
  * **Custom Stopwords Removal:** Xóa bỏ các từ dừng tiếng Anh (NLTK) và **đặc biệt xóa bỏ các từ vựng mang tính nội bộ của Enron** (VD: *enron, ect, hou, re...*) được phát hiện trong bước EDA. Bước này giúp mô hình triệt tiêu hoàn toàn **Domain Bias**.
* **Label Encoding & Data Checkpoint:**
  * **Mã hóa nhãn mục tiêu:** Chuyển đổi thuộc tính nhãn phân loại từ dạng chữ sang dạng số nhị phân: **Spam $\rightarrow$ 1** và **Ham $\rightarrow$ 0** để sẵn sàng cho các thuật toán học có giám sát tính toán toán học.
  * **Xuất dữ liệu trung gian:** Trích lọc và đóng gói dữ liệu sau tiền xử lý thành một file CSV lưu trữ tập trung gồm 4 cột đặc trưng cốt lõi: `Cleaned_Message` (văn bản đã làm sạch), `Message_Length` (độ dài gốc), `Punct_Count` (số dấu câu gốc), và `Label` (nhãn đã mã hóa)

### Feature Engineering

**Mục tiêu:** Chuyển đổi dữ liệu chữ và số liệu thô thành ma trận đặc trưng

* **Feature Creation:**
  * Xây dựng thuộc tính mới `Punctuation_Ratio = Punct_Count / Message_Length` (2 cột đếm này được trích xuất từ văn bản gốc TRƯỚC KHI thực hiện bước làm sạch xóa dấu câu) để bắt hành vi lạm dụng ký tự đặc biệt của lừa đảo
  * **Loại bỏ 2 biến gốc** (`Punct_Count` và `Message_Length`) để chống hiện tượng Đa cộng tuyến và tối ưu hóa bộ nhớ RAM.
* **Data Split:** Chia Train/Test theo tỷ lệ 80/20 trước khi thực hiện Feature Extraction để chống hiện tượng Data Leakage.
* **Feature Extraction:**
  * Sử dụng thuật toán **TF-IDF (Term Frequency - Inverse Document Frequency)** để chuyển đổi cột văn bản thành ma trận trọng số từ vựng (Chỉ `fit_transform` trên tập Train và `transform` trên tập Test).
* **Feature Transformation:**
  * Sử dụng phép biến đổi Logarit (`Log-Transform`) cho cột `Punctuation_Ratio` nhằm giảm hiện tượng lệch phải.
  * Sử dụng `Min-Max Scaling` cho cột Ratio (sau khi log) để ép dữ liệu về khoảng `[0, 1]`, đồng bộ hệ quy chiếu với ma trận TF-IDF.
* **Hợp nhất dữ liệu:** 
  * Ghép nối ma trận thưa TF-IDF và cột Ratio đã scale để tạo thành ma trận đầu vào hoàn chỉnh `X_train_final` và `X_test_final` sẵn sàng nạp vào mô hình phân loại.


### Implement Model From Scratch

Các thuật toán được triển khai hoàn toàn từ đầu nhằm mục tiêu hiểu rõ nguyên lý hoạt động của từng mô hình:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* Naive Bayes

Mỗi mô hình được huấn luyện trên cùng một tập dữ liệu và cùng quy trình tiền xử lý nhằm đảm bảo tính công bằng trong quá trình so sánh.

### Hyperparameter Tuning

Tiến hành tối ưu siêu tham số cho từng mô hình nhằm cải thiện hiệu suất phân loại.

Các siêu tham số được khảo sát bao gồm:

* Logistic Regression:

  * Learning Rate
  * Epochs

* Random Forest:

  * Number of Trees
  * Maximum Depth
  * Minimum Samples Split

* SVM:

  * Learning Rate
  * Lambda Parameter


* Naive Bayes:

  * Alpha

### Scratched Model Evaluation

Các mô hình được đánh giá dựa trên các chỉ số:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* True Positive Rate (TPR - Recall)
* False Positive Rate (FPR - Tỷ lệ báo nhầm)

Sau quá trình đánh giá, tiến hành so sánh hiệu suất giữa các mô hình nhằm xác định thuật toán phù hợp nhất đối với bài toán phân loại Email HAM/SPAM.

## Hướng phát triển

* Mở rộng từ bài toán phân loại nhị phân (HAM/SPAM) sang bài toán phân loại nhiều nhóm Email khác nhau.
