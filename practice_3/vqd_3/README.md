# Chuyên đề Phân cụm Khách hàng Ô tô - Built from Scratch

Dự án này thực hiện quy trình phân cụm học máy không giám sát (Unsupervised Clustering) nhằm phân khúc khách hàng cho một doanh nghiệp sản xuất và phân phối ô tô khi thâm nhập thị trường mới. Dự án được triển khai theo phương pháp **Built from Scratch** (lập trình toàn bộ thuật toán bằng Python thuần và NumPy/Pandas, chỉ dùng thư viện có sẵn để đối chiếu hiệu năng cuối cùng) và được đóng gói trong **một file notebook duy nhất** `final_clustering.ipynb`.

---

##  Định nghĩa bài toán (Define Problem)
 
*   **Bối cảnh**: Một hãng xe hơi muốn thâm nhập vào thị trường mới với các dòng sản phẩm hiện có (P1, P2, P3, P4, P5). Họ cần phân nhóm tập khách hàng tiềm năng tại thị trường mới để tùy chỉnh thiết kế chiến lược tiếp thị, phân phối sản phẩm ô tô phù hợp. `https://www.kaggle.com/datasets/vetrirah/customer/data`
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

---

##  Tiến trình thực hiện trong `final_clustering.ipynb`

### Bước 1: Định nghĩa bài toán (Define Problem)
*   **1. Bối cảnh doanh nghiệp và Đề bài**: Phân tích bối cảnh hãng ô tô thâm nhập thị trường mới, xác định đây là bài toán học máy không giám sát.
*   **2. Định nghĩa toán học của bài toán**: Định nghĩa toán học của Input ($X$) và Output ($C$) gồm $K$ cụm và nhãn nhiễu $-1$ (cho DBSCAN).
*   **3. Cơ sở toán học của 4 thuật toán phân cụm**: Trình bày chi tiết lý thuyết của K-Means, K-Medoids (Lưu ý thiết kế: Euclidean), DBSCAN, Hierarchical (Lance-Williams & Ward Linkage).

### Bước 2: Kiểm tra dữ liệu (Data Collection & Validation)
*   **1. Tải tập dữ liệu thô (Raw Data)** và **2. Xem trước một số dòng dữ liệu**: Nạp và xem sơ bộ cấu trúc 8,068 dòng dữ liệu thô.
*   **3. Kiểm tra kiểu dữ liệu** và **4. Kiểm tra giá trị khuyết thiếu (Missing Values)**: Xác định các cột chứa giá trị thiếu.
*   **5. Kiểm tra dòng trùng lặp (Duplicate Rows)** và **6. Khảo sát sơ bộ các giá trị phân loại**.
*   **7. Mô tả thống kê sơ bộ (Descriptive Statistics)** và **8. Kiểm tra phân phối nhãn mục tiêu (Segmentation Distribution)**.
*   **9. Kiểm tra tính nhất quán logic của dữ liệu (Consistency Checks)**: Phát hiện 137 lỗi Age vs Work_Experience và 518 lỗi Profession vs Graduation.

### Bước 3: Làm sạch dữ liệu (Data Cleaning)
*   **2. Loại bỏ các dòng trùng lặp (Duplicates)**: Loại bỏ các bản ghi ID trùng lặp.
*   **3. Loại bỏ các giá trị mâu thuẫn logic (Sanity Outliers Cleaning)**: Loại bỏ đúng 646 dòng thực sự vi phạm logic (giữ lại các dòng chứa NaN hợp lệ).
*   **4. Khảo sát & Phân tích giá trị khuyết thiếu trước khi xử lý (Missing Values Analysis)**: Rà soát kỹ phân phối trước khi điền khuyết.
*   **5. Thực hiện điền khuyết (Imputation)**: Điền khuyết Median cho biến số học (`Work_Experience = 1.0`, `Family_Size = 2.0`) và Mode cho biến phân loại.
*   **6. Đánh giá độ chệch phân phối dữ liệu (Bias Check - Raw vs. Cleaned Data)**: Đảm bảo phân phối sau làm sạch không bị chệch so với ban đầu.
*   **7. Kiểm tra trực quan giá trị dị biệt (Outliers Visualization)**: Capping/Trimming dữ liệu ngoài ranh giới IQR.
*   **8. Lưu trữ dữ liệu đã làm sạch**: Xuất tập dữ liệu sạch hoàn chỉnh gồm **7,422 dòng**.

### Bước 4: Khám phá dữ liệu (Exploratory Data Analysis - EDA)
*   **2. Phân tích trực quan hóa biến số học (Numerical Features)**: Vẽ phân phối và boxplot của Age, Work_Experience, Family_Size.
*   **3. Phân tích trực quan hóa biến phân loại (Categorical Features)**: Vẽ biểu đồ tần suất cho Gender, Ever_Married, Graduated, Spending_Score, Profession.
*   **4. Kiểm định liên kết phi giám sát toàn diện (Comprehensive Chi-Square & Cramer's V Test Suite)**: Đo đạc độ tương quan phi tuyến giữa các biến phân loại.
*   **5. Khảo sát mối quan hệ đa biến toàn diện (Comprehensive Multivariate Analysis)**: Khảo sát Age theo Ever_Married & Spending_Score, Spending_Score chéo theo tất cả biến phân loại, không gian tương quan của Age/Work_Experience/Family_Size.
*   **6. Tổng hợp các phát hiện quan trọng (Key Insights)**: Đúc kết các phát hiện thống kê nổi bật.

### Bước 5: Xử lý đặc trưng (Feature Engineering)
*   **2. Phân tách tập dữ liệu**: Cô lập nhãn doanh nghiệp riêng biệt trước khi xử lý đặc trưng.
*   **3. Chuẩn hóa dữ liệu số học (Custom Scalers từ Scratch)**: Lập trình lớp `CustomMinMaxScaler`.
*   **4. Mã hóa biến phân loại từ Scratch**: Lập trình lớp `CustomOneHotEncoder`.
*   **5. Giảm chiều dữ liệu PCA từ Scratch (Custom PCA)**: Lập trình thuật toán SVD/Covariance với `ddof=0`.
*   **6. Xây dựng quy trình xử lý đặc trưng và Khảo sát PCA**: Áp dụng Log-transform cho `Work_Experience` và mã hóa chuẩn hóa PCA 2D toàn bộ 7,422 dòng.
*   **7. Trực quan hóa dữ liệu đặc trưng huấn luyện trước khi lưu trữ** và **8. Lưu trữ dữ liệu xử lý đặc trưng đầy đủ**.

### Bước 6.1: KMeans & KMedoids Clustering
*   **2. Hàm tính điểm Silhouette Score**: Lập trình Silhouette Score từ Scratch.
*   **3. Lập trình K-Means từ Scratch**: Triển khai lớp `KMeansScratch`.
*   **4. Tìm số cụm K tối ưu cho K-Means**: Chạy Elbow và Silhouette Score xác định $K=3$.
*   **5. Huấn luyện K-Means với K tối ưu (K=3) & Trực quan hóa 2D**: Trực quan hóa ranh giới cụm 2D và phân tích bán kính phân tán.
*   **6. Lập trình K-Medoids từ Scratch**: Triển khai lớp `KMedoidsScratch` (Euclidean).
*   **7. Tìm số cụm K tối ưu cho K-Medoids**: Xác định $K=4$.
*   **8. Huấn luyện K-Medoids với K tối ưu & Trực quan hóa 2D**: Trực quan hóa 2D và đề xuất phản biện DBSCAN dựa trên ma trận khoảng cách medoid.
*   **9. Lưu trữ mô hình**.

### Bước 6.2: DBSCAN Clustering
*   **2. Lập trình DBSCAN từ Scratch**: Lập trình lớp `DBSCANScratch` loang cụm bằng hàng đợi `deque` ($O(N)$).
*   **3. Cơ sở xác định phạm vi quét Epsilon: Đồ thị K-Distance đa tham chiếu**: Vẽ đồ thị K-Distance với $k = 44$ ($2 \times d$).
*   **4. Dùng K-Means & K-Medoids hướng dẫn thiết lập Epsilon cho DBSCAN**: Lấy khoảng cách L2 trung bình làm cơ sở thiết lập $\epsilon = 1.42$.
*   **5. Tìm kiếm lưới khống chế số cụm dựa trên tập con đại diện**: Quét lưới tìm tham số tối ưu chốt $\epsilon = 1.42$, $MinPts = 70$.
*   **6. Huấn luyện DBSCAN tối ưu & Trực quan hóa 2D đa màu sắc**: Cho ra cấu hình 1 cụm lớn và các điểm nhiễu phân tán.
*   **7. Lưu trữ mô hình**.

### Bước 6.3: Hierarchical Clustering (Agglomerative)
*   **2. Lập trình Hierarchical (Agglomerative) bằng Lance-Williams từ Scratch**: Lớp `AgglomerativeClusteringScratch` hỗ trợ 5 linkages, xây dựng ma trận liên kết đầy đủ ($N-1$ hàng).
*   **3. Kiểm chứng chéo và vẽ Dendrogram cho cả 5 Linkages**: Đánh giá Silhouette Score, ghi nhận Ward Linkage đạt điểm cao nhất.
*   **3.1. Phóng to Dendrogram Ward và xác định K=3 bằng đường cắt động (Dynamic Cut-off)**: Tính toán tự động đường cắt từ ma trận gộp để cắt ra đúng $K=3$ cụm trên tập con 1,500 mẫu.
*   **4. Lưu trữ mô hình**.

### Bước 7.1: Phân tích chân dung khách hàng từ cụm K-Medoids
*   **2. Thống kê đặc trưng số học**: Khảo sát phân bố Age, Work_Experience, Family_Size của 4 cụm K-Medoids trên 6,132 mẫu sạch.
*   **3. Trực quan hóa tỷ lệ các đặc trưng nhân khẩu học theo cụm**: Biểu đồ cột tỷ lệ % cho Gender, Ever_Married, Graduated, Spending_Score.
*   **4. Trực quan hóa đặc trưng nổi trội của các cụm**: So sánh sự khác biệt lớn về Nghề nghiệp và Mức chi tiêu giữa 4 cụm.
*   **5. Định hình chân dung khách hàng mua ô tô (K-Medoids Personas)**: Phác họa chi tiết 4 chân dung khách hàng (Cụm 0: Nam lớn tuổi thành đạt → Lux SUV; Cụm 1: Người trẻ độc thân chi tiêu tối giản; Cụm 2: Nam trung niên đã lập gia đình; Cụm 3: Phụ nữ trung niên tự lập).

### Bước 7.2: Phân tích chân dung khách hàng từ cụm Hierarchical
*   **2. Thống kê đặc trưng số học** và **3. Trực quan hóa nhân khẩu học**: Khảo sát phân bố của 3 cụm Hierarchical (Ward) trên tập con 1,500 mẫu sạch.
*   **5. Định hình chân dung khách hàng mua ô tô (Hierarchical Personas)**: Phác họa chi tiết chân dung 3 nhóm khách hàng chéo theo các biến Hôn nhân, Giới tính và Học vấn.

### Bước 8: Kiểm chứng chéo mô hình (Model Verification)
*   **1. Tải dữ liệu đặc trưng và các nhãn từ Scratch**: Nạp tệp `X_train.csv` và toàn bộ nhãn đã xuất từ các bước trước.
*   **2. Huấn luyện song song mô hình thư viện chuẩn**: Gọi KMeans, DBSCAN, AgglomerativeClustering của Sklearn/SciPy với cùng siêu tham số ($K=3/4$, $\epsilon=1.42$, $MinPts=70$, Ward linkage).
*   **3. Tính toán các chỉ số kiểm tra định lượng (ARI, Inertia, Cost)**: Đo đạc sự trùng khớp giữa Scratch và thư viện.
*   **4. Biểu diễn bảng đối sánh tổng kết (Model Sanity Check Matrix)**: Lập bảng so sánh ARI và sai số giữa Scratch và Thư viện, xác nhận ARI = 1.000000 và trực quan hóa heatmap.

---

##  Cấu trúc thư mục dự án

```text
vqd_3/
├── README.md                           <- Tệp hướng dẫn này
└── final_clustering.ipynb              <- Toàn bộ quy trình phân cụm từ Scratch (Bước 1 → Bước 8)
```
