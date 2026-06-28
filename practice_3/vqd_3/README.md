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
*   **Hàm mục tiêu (Objective/Loss Function)**: Tối thiểu hóa khoảng cách nội cụm (Inertia/WCSS cho KMeans) hoặc tối ưu hóa độ chênh lệch mật độ vùng biên (DBSCAN) và khoảng cách liên kết tối thiểu hóa phương sai (Ward Linkage cho Phân cấp).

---

## 📂 Danh sách các tệp Jupyter Notebook & Tiến trình thực hiện

Dự án đi qua 8 bước tương ứng với cấu trúc tệp chuyên nghiệp:

### 1. `01_introduction.ipynb` (Define Problem)
*   Phân tích đề bài, xác định đây là bài toán học máy không giám sát (Unsupervised Clustering).
*   Định nghĩa toán học chi tiết cho Input ($X$) và các hàm mục tiêu của 3 trường phái phân cụm (Phân hoạch, Mật độ, Phân cấp).

### 2. `02_data_checks.ipynb` (Data Collection & Validation)
*   Kiểm tra cấu trúc dữ liệu thô: kích thước dòng/cột, kiểu dữ liệu đặc trưng (Numerical, Categorical).
*   Rà soát giá trị bị thiếu (Missing Values), các bản ghi trùng lặp và tính toàn vẹn của dữ liệu.

### 3. `03_data_cleaning.ipynb` (Data Cleaning)
*   Xử lý điền khuyết (Imputation) dữ liệu: Điền khuyết độ tuổi bằng trung vị nhóm, điền khuyết các biến danh mục bằng Mode (giá trị phổ biến nhất).
*   Loại bỏ các dòng dữ liệu dị biệt (Outliers) hoặc trùng lặp không hợp lệ để làm sạch dữ liệu.

### 4. `04_eda.ipynb` (Exploratory Data Analysis)
*   Trực quan hóa phân phối độ tuổi, quy mô gia đình bằng biểu đồ Histogram và Boxplot.
*   Phân tích mối tương quan giữa nghề nghiệp, tình trạng kết hôn và mức độ chi tiêu để phát hiện các phân khúc khách hàng tiềm năng sơ bộ.

### 5. `05_feature_engineering.ipynb` (Feature Processing)
*   Áp dụng biến đổi Log Transformation $f(x) = \log(x + 1)$ cho biến `Work_Experience` bị lệch phải mạnh từ Scratch.
*   Lập trình lớp chuẩn hóa dữ liệu số học `CustomMinMaxScaler` từ Scratch đưa các đặc trưng về khoảng $[0, 1]$.
*   Lập trình lớp mã hóa danh mục `CustomOneHotEncoder` từ Scratch cho các nominal features.
*   Lập trình thuật toán giảm chiều `CustomPCA` từ Scratch phục vụ khảo sát trực quan hóa không gian biên cụm 2D.
*   Tự viết logic chia tách tập huấn luyện Train (80%) và tập xác thực Val (20%) có cơ chế đặt hạt giống ngẫu nhiên `seed=42`.

### 6. Huấn luyện thuật toán (Built from Scratch & Training)
*   **`06_kmeans_kmedoids.ipynb`**: Lập trình lớp `KMeansScratch` và `KMedoidsScratch` (khoảng cách Manhattan). Phân tích đồ thị khuỷu tay (Elbow) và Silhouette xác định số cụm tối ưu $K = 3$. Bổ sung tính toán khoảng cách chéo tâm cụm để kiểm chứng độ phủ cụm.
*   **`06_dbscan.ipynb`**: Lập trình lớp `DBSCANScratch`. Sử dụng đồ thị K-Distance ($k=15$) làm cơ sở toán học quét epsilon $\epsilon \in [1.40, 1.50]$. Phân tích cầu mật độ (Density Bridge) để điều chỉnh mật độ tỉ lệ tăng $MinPts = 70$ trên tập Train, thu được 2 cụm rõ rệt kèm nhiễu.
*   **`06_hierarchical.ipynb`**: Lập trình lớp `AgglomerativeClusteringScratch` sử dụng hệ thức cập nhật khoảng cách liên kết **Lance-Williams** hiệu năng cao ($O(N^2)$) hỗ trợ 5 linkages. Vẽ phóng to Dendrogram của phương pháp Ward tối ưu và kẻ đường cắt tại độ cao `y = 25.5` để thu được cấu hình $K = 4$ cụm.

### 7. Đánh giá & Phân tích chân dung (Model Evaluation & Profiling)
*   **`07_evaluation.ipynb`**: Huấn luyện các mô hình đại diện từ Scikit-learn (`KMeans` K=3 và `AgglomerativeClustering` K=4) để đo độ lệch đồng thuận chéo bằng chỉ số ARI và NMI. Trực quan hóa ma trận đối chiếu chéo (Cross-Tabulation Heatmap). Đối chiếu kiểm chứng nhãn thuật toán tự viết Scratch đạt độ khớp ARI xấp xỉ **1.0 (chính xác tuyệt đối)**.
*   **`07_1_kmeans_profiling.ipynb`**: Trực quan hóa chi tiết 3 cụm của KMeans (kèm nhãn % số liệu cụ thể) và phác họa chân dung xe hơi:
    *   *Cụm 0*: Khách hàng trẻ độc thân chi tiêu thấp $\rightarrow$ Dòng xe đô thị hạng A/B giá rẻ, Mini EV.
    *   *Cụm 1*: Phụ nữ trung niên đã kết hôn chi tiêu trung bình/cao $\rightarrow$ Crossover/SUV gia đình cao cấp, xe Hybrid.
    *   *Cụm 2*: Nam giới trung niên đã kết hôn chi tiêu trung bình/cao $\rightarrow$ Sedan hạng sang, SUV 7 chỗ hiệu năng cao.
*   **`07_2_hierarchical_profiling.ipynb`**: Trực quan hóa chi tiết 4 cụm của Hierarchical dựa trên ma trận 2x2 (Hôn nhân & Học vấn) và phác họa chiến lược sản phẩm xe hơi:
    *   *Cụm 0 (Đã kết hôn & Tốt nghiệp)*: Khách hàng lớn tuổi thành đạt $\rightarrow$ Xe sang hạng E/F, Luxury EV.
    *   *Cụm 1 (Chưa kết hôn & Tốt nghiệp)*: Khách hàng trẻ trí thức độc thân $\rightarrow$ Active Crossover, Smart EV.
    *   *Cụm 2 (Đã kết hôn & Chưa tốt nghiệp)*: Khách hàng trung niên MPV $\rightarrow$ MPV gia đình, SUV 7 chỗ đa năng.
    *   *Cụm 3 (Chưa kết hôn & Chưa tốt nghiệp)*: Khách hàng trẻ tối giản $\rightarrow$ Xe Hatchback đô thị cỡ nhỏ, xe cũ chính hãng (CPO).

---

## 🛠️ Cấu trúc thư mục dự án

```text
vqd_3/
├── README.md                           <- Tệp hướng dẫn này
├── 01_introduction.ipynb               <- Định nghĩa bài toán phân cụm ô tô
├── 02_data_checks.ipynb                <- Kiểm tra tính toàn vẹn dữ liệu
├── 03_data_cleaning.ipynb              <- Imputation và loại bỏ outliers
├── 04_eda.ipynb                        <- Phân tích khám phá và tương quan
├── 05_feature_engineering.ipynb        <- Biến đổi đặc trưng và Scalers từ Scratch
├── 06_kmeans_kmedoids.ipynb            <- Huấn luyện KMeans/KMedoids từ Scratch
├── 06_dbscan.ipynb                     <- Huấn luyện DBSCAN từ Scratch
├── 06_hierarchical.ipynb               <- Huấn luyện Hierarchical từ Scratch
├── 07_evaluation.ipynb                 <- Đánh giá và đo lường độ lệch
├── 07_1_kmeans_profiling.ipynb         <- Chân dung 3 cụm KMeans cho hãng xe hơi
└── 07_2_hierarchical_profiling.ipynb   <- Chân dung 4 cụm Hierarchical cho hãng xe hơi
```
