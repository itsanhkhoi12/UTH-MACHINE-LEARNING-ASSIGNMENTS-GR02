# Báo Cáo Bài Tập Thực Hành Môn Học Máy - Nhóm 02

Chào mừng bạn đến với kho lưu trữ mã nguồn bài tập thực hành môn **Học máy (Machine Learning)** của **Nhóm 02** - Trường Đại học Giao thông Vận tải (UTH). Đây là nơi tổng hợp toàn bộ mã nguồn, dữ liệu thử nghiệm và báo cáo kết quả cho các bài tập thực hành trong học phần.

---

## 📊 Bảng Tiến Độ Dự Án

Dưới đây là trạng thái thực hiện các bài tập thực hành lớn trong học kỳ:

| Bài thực hành | Đề tài | Thư mục | Trạng thái | Mô hình tốt nhất | Điểm số cao nhất |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **PRACTICE 1** | **Phân loại Email Spam (Spam Classification)** | [practice_1](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_1) | 🟡 Đang làm | *Naive Bayes / Random Forest* | *Đang cập nhật* |
| **PRACTICE 2** | **Dự đoán Doanh số Sản phẩm (Product Sales)** | [practice_2](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/practice_2) | 🔴 Chưa bắt đầu | *Chưa có* | *Chưa có* |

---

## 📂 Sơ Đồ Cấu Trúc Thư Mục

Dự án được tổ chức một cách khoa học để quản lý dễ dàng giữa mã nguồn thực thi, dữ liệu và các tiện ích dùng chung:

```text
UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/
├── .cursorrules               # Cấu hình vai trò & quy tắc của Agent trên Cursor IDE
├── .clinerules                # Hướng dẫn kỹ thuật và workflow của Agent
├── AGENT_GUIDELINES.md        # Quy trình cộng tác từng bước giữa Người dùng & Agent
├── README.md                  # Báo cáo tổng quan dự án (File này)
├── requirements.txt           # Danh sách thư viện Python cần thiết
├── practice_1/                # BÀI THỰC HÀNH 1: PHÂN LOẠI EMAIL SPAM
│   ├── README.md              # Báo cáo chi tiết kết quả thực hành 1
│   ├── data/                  # Thư mục chứa dữ liệu đầu vào (Spam dataset)
│   ├── models/                # Thư mục lưu trữ các mô hình .pkl đã huấn luyện
│   ├── utils/                 # Thư mục chứa các script tiền xử lý/Cross-validation tùy chỉnh
│   └── *.ipynb                # Các file notebook thực thi (EDA, Train, Eval)
├── practice_2/                # BÀI THỰC HÀNH 2: DỰ ĐOÁN DOANH SỐ SẢN PHẨM
│   ├── README.md              # Báo cáo chi tiết kết quả thực hành 2
│   ├── data/                  # Thư mục chứa dữ liệu doanh số
│   └── *.ipynb                # Các file notebook thực thi
└── utils/                     # Thư mục tiện ích chung cho cả dự án (quản lý model)
    └── model_manager.py       # Module lưu và tải mô hình tự động
```

---

## 🛠️ Hướng Dẫn Cài Đặt Môi Trường

Để chạy các file Jupyter Notebook trong dự án này, vui lòng cài đặt môi trường Python theo các bước sau:

### 1. Khởi tạo môi trường ảo (Khuyến nghị)
Sử dụng `venv` đi kèm với Python:
```bash
python3 -m venv venv
source venv/bin/activate  # Trên macOS/Linux
# Hoặc trên Windows:
# venv\Scripts\activate
```

### 2. Cài đặt các thư viện cần thiết
Cài đặt toàn bộ các thư viện được liệt kê trong [requirements.txt](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/requirements.txt):
```bash
pip install -r requirements.txt
```

---

## 🤖 Hướng Dẫn Dành Cho AI Agent (Antigravity/Cursor/Cline)

Nếu bạn là AI Agent đang hỗ trợ dự án này, hãy đọc kỹ:
1. File cấu hình hoạt động [AGENT_GUIDELINES.md](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/AGENT_GUIDELINES.md) để biết cách thức thực hiện và báo cáo tiến độ.
2. Tuân thủ định dạng viết mã nguồn chuẩn hóa được nêu trong [.cursorrules](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/.cursorrules) và [.clinerules](file:///Users/vqd2k6/Desktop/Học%20máy%20-%20UTH/Pra_1/UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02/.clinerules).
3. Luôn trao đổi với Người dùng bằng **tiếng Việt**.