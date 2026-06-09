import numpy as np

class CustomKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """Khởi tạo bộ chia Stratified K-Fold Cross Validation
        Args:
            n_splits (int): Số lượng fold (thường là 5 hoặc 10)
            shuffle (bool): Có xáo trộn dữ liệu trước khi chia hay không
            random_state (int): Seed ngẫu nhiên để kết quả có thể tái lập
        """
        if n_splits <= 1:
            raise ValueError("Số lượng splits phải lớn hơn 1")
            
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """
        Khác với KFold thường chỉ cần X để đếm độ dài, 
        StratifiedKFold bắt buộc phải nhận y để biết nhãn (Label) mà chia tỷ lệ
        """
        y = np.array(y) if not hasattr(y, 'iloc') else y.values
        n_samples = len(y)
        
        # Tìm các nhãn duy nhất (Ví dụ: 0 và 1)
        unique_classes = np.unique(y)
        
        # Tạo sẵn k mảng rỗng (k thùng) để chứa index
        folds = [[] for _ in range(self.n_splits)]
        
        rng = np.random.RandomState(self.random_state)
        
        # Rải đều từng nhãn vào các thùng
        for cls in unique_classes:
            # Lấy toàn bộ vị trí (index) của nhãn hiện tại
            cls_indices = np.where(y == cls)[0]
            
            # Xáo trộn index của nhãn này nếu có yêu cầu
            if self.shuffle:
                rng.shuffle(cls_indices)
                
            # Rải từng index vào các thùng theo thứ tự (chia lấy dư)
            # Giúp đảm bảo thùng nào cũng có số lượng nhãn cls đồng đều nhất
            for i, idx in enumerate(cls_indices):
                folds[i % self.n_splits].append(idx)
                
        # Trả về kết quả train/val cho từng fold
        for i in range(self.n_splits):
            val_indices = np.array(folds[i])
            
            # Train indices là tổng hợp của tất cả các thùng CÒN LẠI
            train_indices = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            
            yield train_indices, val_indices