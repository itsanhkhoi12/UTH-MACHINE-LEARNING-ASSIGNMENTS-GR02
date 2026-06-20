# Bài Thực Hành 1: Phân Loại Email Spam (Spam Email Classification)

Bài thực hành này tập trung vào việc phát triển hệ thống phân loại thư rác (Spam) và thư thường (Ham) sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) kết hợp với các thuật toán Học máy phổ biến. Dự án thử nghiệm cả các mô hình tự viết từ đầu (From Scratch) và các mô hình từ thư viện chuẩn (Scikit-learn) để đối sánh hiệu năng.

---

## 📂 Danh Sách Các Bước Thực Hiện (Notebooks)

Tiến trình thực hiện được chia nhỏ thành các file Notebook theo quy trình chuẩn:
1. [01-02-eda-data-preprocessing.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_1/01-02-eda-data-preprocessing.ipynb): Phân tích khám phá dữ liệu (EDA) và làm sạch văn bản (loại bỏ HTML, ký tự đặc biệt, chuyển chữ thường, tokenization, loại bỏ stop words).
2. [03-feature-engineering.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_1/03-feature-engineering.ipynb): Trích chọn đặc trưng sử dụng TF-IDF Vectorizer để chuyển đổi văn bản thành các vector số.
3. [04-train-logistic-regression.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_1/04-train-logistic-regression.ipynb): Xây dựng và đánh giá mô hình Hồi quy Logistic (Sklearn vs Scratch).
4. [05-train-random-forest.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_1/05-train-random-forest.ipynb): Xây dựng và đánh giá mô hình Random Forest (Scratch).
5. [06-train-svm.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_1/06-train-svm.ipynb): Xây dựng và đánh giá mô hình SVM phân loại tuyến tính (Sklearn vs Scratch).
6. [07-train-navie-bayes.ipynb](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_1/07-train-navie-bayes.ipynb): Xây dựng và đánh giá mô hình Multinomial Naive Bayes (Sklearn vs Scratch).

---

## 📊 Chỉ Số Đánh Giá (Metrics)

Vì bài toán phân loại email spam thường gặp hiện tượng mất cân bằng dữ liệu (số lượng thư thường nhiều hơn nhiều so với thư rác), việc sử dụng chỉ số **Accuracy** là chưa đủ. Dự án tập trung tối ưu hóa các chỉ số sau:
- **Precision (Độ chính xác)**: Tỷ lệ thư được dự đoán là Spam thực sự là Spam. Chỉ số này cần cực kỳ cao để tránh việc phân loại nhầm thư quan trọng (Ham) vào thư mục Spam.
- **Recall (Độ phủ)**: Tỷ lệ thư Spam thực tế được mô hình phát hiện ra.
- **F1-Score**: Điểm trung bình điều hòa giữa Precision và Recall.

---

## 🏆 Bảng So Sánh Hiệu Năng Các Mô Hình

Dưới đây là bảng kết quả chạy thực tế trên tập dữ liệu kiểm thử (Test Set):

| Mô hình | Thuật toán | Loại | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | Multinomial NB | Scikit-learn | **98.29%** | **98.48%** | **98.16%** | **98.32%** |
| | Multinomial NB | From Scratch | **98.29%** | **98.48%** | **98.16%** | **98.32%** |
| **SVM** | Linear SVM | Scikit-learn | 98.36% | 97.86% | 98.69% | 98.27% |
| | Linear SVM | From Scratch | 88.48% | 98.61% | 76.74% | 86.31% |
| **Logistic Regression** | Logistic Regression | From Scratch | 83.73% | 90.79% | 73.03% | 80.95% |
| | Logistic Regression | Scikit-learn | 83.17% | 98.38% | 65.51% | 78.65% |
| **Random Forest** | Random Forest | From Scratch | 77.07% | 97.18% | 53.09% | 68.66% |

> [!TIP]
> **Nhận xét kết quả:** Mô hình **Multinomial Naive Bayes** (cả bản cài đặt từ đầu và bản dùng thư viện) cho hiệu năng tốt và đồng đều nhất với F1-Score đạt **98.32%**. Mô hình **Linear SVM từ thư viện** cũng cho kết quả rất tốt (F1-Score **98.27%**).

---

## 📦 Lưu Trữ Mô Hình (Models Serialization)

Các mô hình được huấn luyện tốt nhất và phục vụ cho việc dự đoán sau này đã được đóng gói và lưu tại thư mục [practice_1/models](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_1/models):
- `Linear_SVM_Scratch.pkl`
- `Logistic_Regression.pkl`
- `Naive_Bayes_Scratch.pkl`
- `Random_Forest_Scratch.pkl`
