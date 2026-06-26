from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent 
DEFAULT_MODEL_DIR = BASE_DIR / 'models'

def generate_report(model_dir=None, X_test=None, y_test=None, plot_cm=False, task_type='classification'):
    target_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    results = []
    cm_list = []
    model_names_for_cm = []
    
    if not target_dir.is_dir():
        print(f"Folder {target_dir.resolve()} không tồn tại")
        return pd.DataFrame()
        
    for file_path in target_dir.glob('*.pkl'):
        if file_path.name.startswith(('onehot', 'standard', 'tfidf', 'vectorizer')):
            continue
            
        try:
            package = joblib.load(file_path)
            model = package.get('model')
            model_name = package.get('model_name', 'Unknown')
            
            best_params = package.get('best_params', {})
            if isinstance(best_params, dict) and best_params:
                formatted_params = ", ".join([f"{k}={v}" for k, v in best_params.items()])
            else:
                formatted_params = "Mặc định (Default)"

            row = {
                'Mô hình': model_name,
                'Tham số tốt nhất': formatted_params
            }
            
            if X_test is not None and y_test is not None and model is not None:
                y_pred = model.predict(X_test)
                y_true = np.array(y_test).flatten()
                
                if task_type == 'classification':
                    cm = confusion_matrix(y_true, y_pred)
                    # ravel() tách mảng 2x2 thành 4 biến
                    tn, fp, fn, tp = cm.ravel() 
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    row['TPR (Độ nhạy)'] = round(tpr, 4)
                    row['FPR (Báo nhầm)'] = round(fpr, 4)
                    
                    if plot_cm:
                        cm_list.append(cm)
                        model_names_for_cm.append(model_name)
                
                elif task_type == 'regression':
                    pass 

            row.update(package.get('metrics', {}))
            results.append(row)
            
        except Exception as e:
            print(f"-> Skip file {file_path.name}: Lỗi quá trình đánh giá - {e}")
                
    if task_type == 'classification' and plot_cm and cm_list:
        n_models = len(cm_list)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1: axes = [axes]
        
        for ax, cm, name in zip(axes, cm_list, model_names_for_cm):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Ham (0)', 'Spam (1)'],
                        yticklabels=['Ham (0)', 'Spam (1)'])
            ax.set_title(f'{name}')
            ax.set_xlabel('Dự đoán (Predicted)')
            ax.set_ylabel('Thực tế (True)')
            
        plt.tight_layout()
        plt.show()

    df = pd.DataFrame(results)
    if df.empty: return df

    if 'F1_Score' in df.columns:
        df = df.sort_values(by='F1_Score', ascending=False)
    elif 'MAE' in df.columns:
        df = df.sort_values(by='MAE', ascending=True)
    elif 'Accuracy' in df.columns:
        df = df.sort_values(by='Accuracy', ascending=False)
    
    cols = [c for c in df.columns if c != 'Tham số tốt nhất'] + ['Tham số tốt nhất']
    return df[cols].reset_index(drop=True)