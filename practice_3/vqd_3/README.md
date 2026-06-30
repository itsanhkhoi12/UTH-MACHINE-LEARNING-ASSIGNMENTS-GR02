# Chuyên đề Phân cụm Khách hàng Ô tô - Built from Scratch

Dự án này thực hiện quy trình phân cụm học máy không giám sát (Unsupervised Clustering) nhằm phân khúc khách hàng cho một doanh nghiệp sản xuất và phân phối ô tô khi thâm nhập thị trường mới. Dự án được triển khai theo phương pháp **Built from Scratch** (lập trình toàn bộ thuật toán bằng Python thuần và NumPy/Pandas, chỉ dùng thư viện có sẵn để đối chiếu hiệu năng cuối cùng).

---

## 🎯 Định nghĩa bài toán (Define Problem)

*   **Bối cảnh**: Một hãng xe hơi muốn thâm nhập vào thị trường mới với các dòng sản phẩm hiện có (P1, P2, P3, P4, P5). Họ cần phân nhóm tập khách hàng tiềm năng tại thị trường mới để tùy chỉnh thiết kế chiến lược tiếp thị, phân phối sản phẩm ô tô phù hợp.
*   **Input ($X$)**: Các đặc trưng nhân khẩu học và tiêu dùng của khách hàng (gồm 9 cột dữ liệu đầu vào):
    *   `Gender` (Giới tính): *Male* (Nam) hoặc *Female* (Nữ).
    *   `Ever_Married` (Đã từng kết hôn): *Yes* (Đã kết hôn) hoặc *No* (Độc thân).
    *   `Age` (Tuổi): Biến số học liên tục biểu diễn tuổi của khách hàng.
    *   `Graduated` (Tốt nghiệp): *Yes* (Đã tốt nghiệp Đại học) hoặc *No* (Chưa tốt nghiệp).
    *   `Profession` (Nghề nghiệp): Gồm các nhóm nghề chính (*Artist, Healthcare, Engineer, Lawyer, Executive, Doctor, Entertainment, Homemaker, Marketing*).
    *   `Work_Experience` (Kinh nghiệm làm việc): Số năm kinh nghiệm tích lũy (biến số học chứa nhiều giá trị 0 và lệch phải).
    *   `Spending_Score` (Mức chi tiêu): Phân lớp chi tiêu của khách hàng (*Low - Thấp, Average - Trung bình, High - Cao*).
    *   `Family_Size` (Quy mô gia đình): Số lượng thành viên trong gia đình (biến số học).
    *   `Var_1` (Mã phân nhóm ẩn danh): Biến định danh phân khúc nội bộ phục vụ bảo mật GDPR (*Cat_1* đến *Cat_7*).
*   **Triết lý thiết kế hệ thống**:
    1.  **Bảo toàn mật độ không gian**: Không phân chia Train/Val để giữ toàn vẹn phân bố phân cụm thực tế. Kích thước tập dữ liệu sau làm sạch là **7,422 mẫu**.
    2.  **Đồng nhất không gian hình học**: Thuật toán K-Medoids được đồng nhất sử dụng khoảng cách Euclidean thay vì Manhattan để so sánh công bằng trên không gian giảm chiều PCA 2D.

---

## 📂 Danh sách các tệp Jupyter Notebook & Tiến trình thực hiện

### 1. `01_introduction.ipynb` (Bước 1: Định nghĩa bài toán)
*   **1. Bối cảnh doanh nghiệp và Đề bài**: Phân tích bối cảnh hãng ô tô thâm nhập thị trường mới, xác định đây là bài toán học máy không giám sát.
*   **3. Định nghĩa toán học của bài toán**: Định nghĩa toán học của Input ($X$) và Output ($C$) gồm $K$ cụm và nhãn nhiễu $-1$ (cho DBSCAN).
*   **4. Cơ sở toán học của 4 thuật toán phân cụm**: Trình bày chi tiết lý thuyết của K-Means, K-Medoids (Lưu ý thiết kế: Euclidean), DBSCAN, Hierarchical (Lance-Williams & Ward Linkage).

### 2. `02_data_checks.ipynb` (Bước 2: Kiểm tra dữ liệu)
*   **2. Kiểm tra thông tin chung và cấu trúc dữ liệu**: Rà soát cấu trúc dữ liệu thô (8,068 dòng).
*   **3. Kiểm tra tính toàn vẹn dữ liệu (Missing Values)**: Phát hiện các cột chứa giá trị khuyết thiếu.
*   **7. Đánh giá liên kết đặc trưng bằng hệ số Cramer's V Test**: Đo đạc độ tương quan phi tuyến giữa các biến phân loại.
*   **10. Kiểm tra tính nhất quán logic của dữ liệu (Consistency Checks)**:
    *   *10.1. Age vs Work_Experience*: Phát hiện 137 lỗi logic ($Age - Work\_Experience < 15$).
    *   *10.2. Profession vs Graduation*: Phát hiện 518 lỗi logic (Bác sĩ/Luật sư chưa tốt nghiệp).
    *   *10.3. Ever Married vs Spending Score*: Không phát hiện lỗi logic (100% độc thân chi tiêu thấp).

### 3. `03_data_cleaning.ipynb` (Bước 3: Làm sạch dữ liệu)
*   **2. Loại bỏ các bản ghi trùng lặp (Duplicates)**: Loại bỏ các bản ghi ID trùng lặp.
*   **3. Xử lý loại bỏ mâu thuẫn logic dữ liệu**: Loại bỏ đúng 646 dòng thực sự vi phạm logic (giữ lại các dòng chứa NaN hợp lệ).
*   **5. Thiết lập quy tắc điền khuyết (Imputation) từ tập huấn luyện**: Điền khuyết Median cho biến số học (`Work_Experience = 1.0`, `Family_Size = 2.0`) và Mode cho biến phân loại.
*   **7. Xử lý giá trị ngoại lệ (Outliers) bằng phương pháp IQR**: Capping/Trimming dữ liệu ngoài ranh giới IQR.
*   **8. Lưu trữ dữ liệu sau khi làm sạch**: Xuất tập dữ liệu sạch hoàn chỉnh gồm **7,422 dòng**.

### 4. `04_eda.ipynb` (Bước 4: Phân tích khám phá dữ liệu)
*   **2. Phân tích trực quan hóa biến số học**: Vẽ phân phối và boxplot của Age, Work_Experience, Family_Size.
*   **4. Kiểm định liên kết phi giám sát toàn diện**: Áp dụng kiểm định Chi-Square & Cramer's V giữa các biến phân loại.
*   **5. Khảo sát mối quan hệ đa biến toàn diện**: 
    *   *5.1. Age theo Ever_Married và Spending_Score*.
    *   *5.2. Mối quan hệ giữa Spending_Score và tất cả các biến phân loại*.
    *   *5.3. Khảo sát không gian tương quan của Age, Work_Experience, Family_Size*.
    *   *5.4. Khai phá sâu các bất thường thống kê và các tổ hợp chập 2 quan trọng*.

### 5. `05_feature_engineering.ipynb` (Bước 5: Xử lý đặc trưng)
*   **2. Phân tách tập dữ liệu**: Cô lập nhãn doanh nghiệp riêng biệt trước khi xử lý đặc trưng.
*   **4. Tự viết bộ chuẩn hóa dữ liệu số học (Custom Scalers từ Scratch)**: Lập trình lớp `CustomMinMaxScaler`.
*   **5. Tự viết bộ mã hóa biến phân loại từ Scratch**: Lập trình lớp `CustomOneHotEncoder`.
*   **6. Tự viết lớp Giảm chiều dữ liệu PCA từ Scratch (Custom PCA)**: Lập trình thuật toán SVD/Covariance với `ddof=0`.
*   **7. Xây dựng quy trình xử lý đặc trưng và Khảo sát PCA**: Áp dụng Log-transform cho `Work_Experience` và mã hóa chuẩn hóa PCA 2D toàn bộ 7,422 dòng.

### 6. `06_kmeans_kmedoids.ipynb` (Bước 6.1: KMeans & KMedoids Clustering)
*   **3. Lập trình K-Means từ Scratch** và **6. Lập trình K-Medoids từ Scratch**: Triển khai các lớp `KMeansScratch` và `KMedoidsScratch` (Euclidean).
*   **4. Tìm số cụm K tối ưu cho K-Means** và **7. Tìm số cụm K tối ưu cho K-Medoids**: Chạy Elbow và Silhouette Score độc lập xác định $K=3$ cho KMeans và $K=4$ cho K-Medoids.
*   **5. Huấn luyện & Trực quan hóa 2D**: Trực quan hóa ranh giới cụm 2D và phân tích bán kính phân tán cụm.
*   **8.2. Nhận xét phản biện học thuật cho DBSCAN**: Đưa ra tiền đề phản bác mật độ của DBSCAN dựa trên bán kính phân tán và ma trận khoảng cách medoid.

### 7. `06_dbscan.ipynb` (Bước 6.2: DBSCAN Clustering)
*   **3. Lập trình DBSCAN từ Scratch**: Lập trình lớp `DBSCANScratch` loang cụm bằng hàng đợi `deque` ($O(N)$).
*   **4. Cơ sở xác định phạm vi quét Epsilon**: Vẽ đồ thị K-Distance với $k = 44$ ($2 \times d$).
*   **5. Dùng K-Means & K-Medoids hướng dẫn thiết lập Epsilon**: Lấy khoảng cách L2 trung bình làm cơ sở thiết lập $\epsilon = 1.42$.
*   **6. Tìm kiếm lưới khống chế số cụm**: Quét lưới tìm tham số tối ưu chốt $\epsilon = 1.42$, $MinPts = 70$.
*   **7. Huấn luyện DBSCAN tối ưu & Trực quan hóa 2D**: Cho ra cấu hình 1 cụm lớn và các điểm nhiễu phân tán.

### 8. `06_hierarchical.ipynb` (Bước 6.3: Hierarchical Clustering)
*   **3. Lập trình Hierarchical (Agglomerative) bằng Lance-Williams từ Scratch**: Lớp `AgglomerativeClusteringScratch` hỗ trợ 5 linkages.
*   **4. Kiểm chứng chéo và vẽ Dendrogram cho cả 5 Linkages**: Đánh giá Silhouette Score, ghi nhận Ward Linkage đạt điểm cao nhất (0.1999).
*   **4.2. Phóng to Dendrogram Ward và xác định K=4 bằng đường cắt động (Dynamic Cut-off)**: Tính toán tự động đường cắt từ ma trận gộp để cắt ra đúng $K=4$ cụm trên tập con 1,000 mẫu.

### 9. `07_evaluation.ipynb` (Bước 7: Đánh giá mô hình - External Evaluation)
*   **2. Huấn luyện các mô hình từ Scikit-learn làm đại diện**: Huấn luyện KMeans K=3 và Hierarchical K=4 của thư viện để làm mốc đối chiếu.
*   **3. Tính toán độ lệch và tương đồng giữa 2 kết quả phân cụm**: Sử dụng ARI và NMI so sánh độ đồng thuận chéo giữa KMeans K=3 và Hierarchical K=4.
*   **4. Đánh giá ngoài đối chiếu với nhãn phân khúc của doanh nghiệp (External Evaluation)**: Đo đạc chỉ số ARI/NMI so với nhãn gốc (`y_train.csv`).
*   **5. Đối chiếu kiểm chứng thuật toán viết từ Scratch**: So sánh nhãn Scratch vs Sklearn trên tập đầy đủ đạt độ khớp tuyệt đối (ARI ≈ 1.0).

### 10. `07_model_verification.ipynb` (Bước 7: Kiểm chứng chéo mô hình - Sanity Check Matrix)
*   **2. Huấn luyện song song mô hình thư viện chuẩn**: Gọi KMeans, DBSCAN, AgglomerativeClustering của Sklearn/SciPy với cùng siêu tham số ($K=4$, $\epsilon=1.42$, $MinPts=70$, Ward linkage).
*   **4. Biểu diễn bảng đối sánh tổng kết (Model Sanity Check Matrix)**: Lập bảng so sánh ARI và sai số (Inertia, số lượng nhiễu) giữa Scratch và Thư viện. Xác nhận sự trùng khớp tuyệt đối (ARI = 1.000000) và trực quan hóa heatmap.

### 11. `07_1_kmedoids_profiling.ipynb` (Bước 7.1: Phân tích chân dung cụm K-Medoids)
*   **2. Thống kê đặc trưng số học** và **3. Trực quan hóa nhân khẩu học**: Khảo sát phân bố của 4 cụm K-Medoids trên 7,422 mẫu sạch.
*   **5. Định hình chân dung khách hàng mua ô tô (K-Medoids Personas)**: Phác họa chi tiết đặc trưng và đề xuất dòng xe phù hợp (Cụm 0: Nam lớn tuổi thành đạt $\to$ Lux SUV; Cụm 1: Người trẻ mới đi làm $\to$ Hatchback/Mini EV; Cụm 2: Phụ nữ trung niên thành đạt $\to$ Hybrid/SUV; Cụm 3: Người độc thân tự lập $\to$ C-SUV/Sedan cỡ C).

### 12. `07_2_hierarchical_profiling.ipynb` (Bước 7.2: Phân tích chân dung cụm Hierarchical)
*   **2. Thống kê đặc trưng số học** và **3. Trực quan hóa nhân khẩu học**: Khảo sát phân bố của 4 cụm Hierarchical trên tập con 1,000 mẫu sạch.
*   **5. Định hình chân dung khách hàng mua ô tô (Hierarchical Personas)**: Phác họa chi tiết chân dung khách hàng chéo theo các biến Hôn nhân, Giới tính và Học vấn.

---

## 🛠️ Cấu trúc thư mục dự án

```text
vqd_3/
├── README.md                           <- Tệp hướng dẫn này
├── 01_introduction.ipynb               <- Định nghĩa toán học bài toán
├── 02_data_checks.ipynb                <- Kiểm tra chất lượng dữ liệu thô
├── 03_data_cleaning.ipynb              <- Tiền xử lý, lọc lỗi logic và điền khuyết
├── 04_eda.ipynb                        <- Phân tích trực quan hóa khám phá
├── 05_feature_engineering.ipynb        <- Log-transform, MinMaxScaler, One-Hot và PCA từ Scratch
├── 06_kmeans_kmedoids.ipynb            <- Huấn luyện KMeans (K=3) & KMedoids (K=4) từ Scratch
├── 06_dbscan.ipynb                     <- Huấn luyện DBSCAN (eps=1.42, MinPts=70) từ Scratch
├── 06_hierarchical.ipynb               <- Huấn luyện Agglomerative Lance-Williams từ Scratch
├── 07_evaluation.ipynb                 <- Đánh giá chéo và đánh giá ngoài vs nhãn doanh nghiệp
├── 07_model_verification.ipynb         <- So khớp nhãn Scratch vs Scikit-learn (ARI = 1.0)
├── 07_1_kmedoids_profiling.ipynb       <- Định hình chân dung 4 cụm K-Medoids cho hãng xe
└── 07_2_hierarchical_profiling.ipynb   <- Định hình chân dung 4 cụm Hierarchical cho hãng xe
```
