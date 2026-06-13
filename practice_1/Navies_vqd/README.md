# Dự án Phân loại Email Spam với Thuật toán Naive Bayes (Thành viên: vqd)

Thư mục `Navies_vqd` chứa toàn bộ giải pháp hoàn chỉnh và độc lập cho bài toán phân loại Email Spam/Ham trên tập dữ liệu Enron, sử dụng mô hình **Multinomial Naive Bayes** viết từ đầu (From Scratch) đối chiếu với thư viện chuẩn **Scikit-learn**.

Giải pháp được bản địa hóa hoàn toàn bằng **tiếng Việt**, cấu trúc mô-đun hóa tối ưu và áp dụng các cải tiến xử lý đặc trưng văn bản chuyên biệt.

---

## 📂 Danh sách các tệp tin trong thư mục `Navies_vqd`

Hệ thống được tổ chức thành **8 bước notebook tuyến tính** giúp người đọc dễ dàng theo dõi từ khâu lý thuyết, xử lý dữ liệu cho đến đánh giá mô hình chuyên sâu:

1. 📓 **`01_introduction.ipynb`**: Giới thiệu tổng quan về dự án, cấu trúc thư mục, lý thuyết toán học của mô hình Naive Bayes và Laplace Smoothing.
2. 📓 **`02_data_checks.ipynb`**: Kiểm tra chất lượng dữ liệu thô (kiểm tra kiểu dữ liệu, xác định tỷ lệ khuyết thiếu của cột tiêu đề và nội dung, kiểm tra phân phối mất cân bằng nhãn).
3. 📓 **`03_data_cleaning.ipynb`**: Làm sạch dữ liệu (loại bỏ trùng lặp, chuẩn hóa cột nhãn thành nhị phân, lọc bỏ 51 dòng trống hoàn toàn thiếu cả tiêu đề và nội dung, lưu dữ liệu sạch trung gian).
4. 📓 **`04_eda.ipynb`**: Phân tích khám phá dữ liệu (vẽ biểu đồ phân phối từ/độ dài ký tự, trích xuất 20 từ đơn và cụm 2 từ phổ biến nhất, thống kê email không tiêu đề, và phân tích tần suất ký tự đặc biệt `!` và `$`).
5. 📓 **`05_text_preprocessing.ipynb`**: Tiền xử lý NLP chuyên sâu (làm sạch HTML/URL, tách từ, loại bỏ stop words, lemmatization) và tự viết bộ trích xuất đặc trưng **TF-IDF từ đầu** (`TFIDFVectorizerFromScratch`).
6. 📓 **`06_model_training.ipynb`**: Xây dựng thuật toán **Multinomial Naive Bayes từ đầu** (`NaiveBayesClassifierFromScratch`), huấn luyện mô hình và thực hiện GridSearchCV tối ưu hóa tham số làm mịn `alpha`.
7. 📓 **`07_evaluation.ipynb`**: Đánh giá chi tiết mô hình (Confusion Matrix, vẽ ROC/AUC tự viết từ đầu, phân tích độ nhạy của `alpha` đến các chỉ số dự đoán và in báo cáo tổng hợp).
8. 📓 **`08_conclusion.ipynb`**: Kết luận dự án, chỉ ra các phát hiện chính, các hạn chế và đề xuất cấu hình tối ưu để triển khai thực tế.
9. 📓 **`09_compare_preprocessing.ipynb`**: Đánh giá và so sánh trực tiếp hiệu quả của quy trình tiền xử lý tự xây dựng (giữ lại các token đặc trưng như `__exclamation__`, `__dollar__` và bảo toàn 33,665 mẫu) so với quy trình tiền xử lý của nhóm phát triển chung ở thư mục `models` bên ngoài (chỉ giữ lại 27,383 mẫu và xóa sạch dấu câu). Kết quả đối chiếu chứng minh giải pháp tự xây dựng giúp tăng tỷ lệ bắt giữ Spam (TPR tăng +0.63%) và giảm mạnh tỷ lệ báo động giả (FPR giảm -0.68%, tương đương với mức giảm 30.2% lượng thư thường bị nhận nhầm thành spam).
10. 🌐 **`sample_emails.html`**: Trình khám phá email tương tác (Email Explorer) được thiết kế hiện đại (Dark Mode, font Outfit, bộ lọc JavaScript), cho phép xem trực quan nội dung chi tiết của **30 email mẫu** (15 Ham và 15 Spam) trích xuất từ tập dữ liệu sạch.
11. 🐍 **`run_notebooks.py`**: Script Python giúp chạy tự động và kiểm thử tuần tự toàn bộ pipeline các notebook từ bước 02 đến bước 07 và bước 09 chỉ với một lệnh duy nhất.

---

## 🌟 Các đặc trưng nổi bật và cải tiến kỹ thuật

* **Tự viết thuật toán từ đầu (From Scratch)**: Tự phát triển hoàn chỉnh lớp trích xuất đặc trưng TF-IDF và mô hình phân loại Multinomial Naive Bayes (có làm mịn Laplace và thuật toán Softmax ổn định số học khi tính toán xác suất dự đoán), đạt độ chính xác tương đương tuyệt đối (khớp 1-1 từng số thập phân) với thư viện Scikit-learn.
* **Đồng bộ hóa quy trình dữ liệu sạch**: Khắc phục lỗi bất đồng bộ của các phiên bản cũ (vốn tải lại file thô ở Bước 5 làm vô hiệu hóa bước làm sạch ở Bước 3). Hiện tại, toàn bộ pipeline đọc nhất quán tệp sạch `enron_cleaned.csv`.
* **Chuẩn hóa đặc trưng đặc thù cho email Spam**:
  * Các con số được quy về token `__number__` để kiểm soát mật độ số (tần suất số trong email Spam rất cao) và tránh bùng nổ từ vựng.
  * Các chuỗi dấu chấm than liên tiếp được mã hóa thành token `__exclamation__` (EDA chứng minh dấu chấm than xuất hiện ở Spam gấp gần 3 lần so với Ham).
  * Ký tự tiền tệ được mã hóa thành token `__dollar__` giúp mô hình nắm bắt tốt các email mời chào tài chính/quảng cáo.
* **Độc lập và an toàn (Project Isolation)**: Toàn bộ pipeline từ tiền xử lý đến lưu trữ đặc trưng và mô hình được cô lập hoàn toàn trong thư mục `Navies_vqd/data/`, đảm bảo không bị lẫn lộn dữ liệu hay lỗi code khi làm việc nhóm chung trên repository.

---

## 📊 Kết quả hiệu năng tối ưu trên tập kiểm thử

Sau khi làm sạch dữ liệu và tối ưu hóa siêu tham số bằng Cross-Validation (`alpha = 0.001`), cả mô hình tự viết (From Scratch) và Scikit-learn đều cho kết quả khớp nhau hoàn hảo:

| Chỉ số đánh giá | Mô hình From Scratch | Mô hình Scikit-learn (MultinomialNB) | Trạng thái đối chiếu |
| :--- | :---: | :---: | :---: |
| **Accuracy (%)** | **98.2920%** | **98.2920%** | Khớp hoàn toàn |
| **Precision (%)** | **98.4764%** | **98.4764%** | Khớp hoàn toàn |
| **Recall (%)** | **98.1600%** | **98.1600%** | Khớp hoàn toàn |
| **F1-Score (%)** | **98.3180%** | **98.3180%** | Khớp hoàn toàn |
| **TPR (%)** | **98.1600%** | **98.1600%** | Khớp hoàn toàn |
| **FPR (%)** | **1.5715%** | **1.5715%** | Khớp hoàn toàn |
| **ROC AUC** | **0.998312** | **0.998312** | Khớp hoàn toàn |

---

## 🚀 Hướng dẫn chạy thử nghiệm

Bạn có thể mở và thực thi tuần tự các notebook từ `01` đến `08` bằng Jupyter Notebook hoặc VS Code Jupyter Extension.

Hoặc chạy script kiểm thử tự động toàn bộ pipeline (02-07) trong nền từ thư mục root của dự án bằng lệnh:
```bash
python3 "practice_1/Navies_vqd/run_notebooks.py"
```
*(Lưu ý: Script kiểm thử tự động cấu hình backend `matplotlib.use('Agg')` để chạy chế độ ẩn, không mở GUI làm treo tiến trình).*
