import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add workspace root to sys.path so pickle can find the custom classes
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

def generate_dashboard():
    print("Starting Threshold Selector Dashboard Generator...")
    
    # 1. Paths configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    data_dir = os.path.join(current_dir, "data", "ready_for_train")
    csv_path = os.path.join(current_dir, "data", "preprocessed", "enron_cleaned.csv")
    output_html_path = os.path.join(current_dir, "threshold_selector.html")
    
    # Check if necessary files exist
    required_files = [
        os.path.join(data_dir, "clf_scratch.pkl"),
        os.path.join(data_dir, "X_test_tfidf.pkl"),
        os.path.join(data_dir, "y_test.pkl"),
        csv_path
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            return
            
    # 2. Load models and vectorized data
    print("Loading model and test data...")
    with open(os.path.join(data_dir, "clf_scratch.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(data_dir, "X_test_tfidf.pkl"), "rb") as f:
        X_test_tfidf = pickle.load(f)
    with open(os.path.join(data_dir, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
        
    # Predict spam probabilities (index 1 is 'spam')
    print("Predicting probabilities...")
    y_proba = clf.predict_proba(X_test_tfidf)[:, 1]
    
    # 3. Load raw emails and recreate the test split
    print("Recreating train-test split for raw texts...")
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].fillna('')
    df['Subject'] = df['Subject'].fillna('(No Subject)')
    
    # Same split parameters as 05_text_preprocessing.ipynb
    y_labels = df['label'].values
    X_raw_texts = df['text'].values
    X_subjects = df['Subject'].values
    
    _, test_raw_texts, _, test_labels = train_test_split(
        X_raw_texts, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    _, test_subjects, _, _ = train_test_split(
        X_subjects, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # Double-check alignment
    assert len(test_labels) == len(y_test), "Test split length mismatch"
    assert np.all(test_labels == y_test), "Test labels mismatch. Split alignment failed!"
    print(f"Successfully aligned test split. Total test samples: {len(y_test)}")
    
    # 4. Precompute metrics for thresholds 0.00 to 1.00
    print("Precomputing metrics for all thresholds...")
    thresholds = np.linspace(0.0, 1.0, 101)
    metrics_data = []
    
    y_test_arr = np.array(y_test)
    
    for t in thresholds:
        # Predict based on current threshold
        y_pred = np.where(y_proba >= t, 'spam', 'ham')
        
        # Calculate Confusion Matrix
        tp = int(np.sum((y_test_arr == 'spam') & (y_pred == 'spam')))
        fp = int(np.sum((y_test_arr == 'ham') & (y_pred == 'spam')))
        tn = int(np.sum((y_test_arr == 'ham') & (y_pred == 'ham')))
        fn = int(np.sum((y_test_arr == 'spam') & (y_pred == 'ham')))
        
        # Calculate Metrics
        total = len(y_test_arr)
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        metrics_data.append({
            "threshold": round(float(t), 2),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "fpr": round(fpr * 100, 2)
        })
        
    # 5. Select representative emails for display
    print("Selecting representative emails for Inspector...")
    selected_indices = set()
    
    # Indices lists based on categories (using threshold 0.5 as baseline reference)
    y_pred_05 = np.where(y_proba >= 0.5, 'spam', 'ham')
    
    # High Confidence Ham (true ham, very low prob)
    high_conf_ham = np.where((y_test_arr == 'ham') & (y_pred_05 == 'ham') & (y_proba < 0.01))[0]
    # High Confidence Spam (true spam, very high prob)
    high_conf_spam = np.where((y_test_arr == 'spam') & (y_pred_05 == 'spam') & (y_proba > 0.99))[0]
    # Borderline (prob between 0.40 and 0.60)
    borderline = np.where((y_proba >= 0.40) & (y_proba <= 0.60))[0]
    # False Positives at 0.5 (actual ham, predicted spam)
    false_positives = np.where((y_test_arr == 'ham') & (y_pred_05 == 'spam'))[0]
    # False Negatives at 0.5 (actual spam, predicted ham)
    false_negatives = np.where((y_test_arr == 'spam') & (y_pred_05 == 'ham'))[0]
    
    # Sample from each category (aim for ~30-40 of each, or all if less)
    np.random.seed(42)
    
    def add_samples(indices_array, limit=40):
        if len(indices_array) == 0:
            return
        sampled = np.random.choice(indices_array, min(limit, len(indices_array)), replace=False)
        for idx in sampled:
            selected_indices.add(int(idx))
            
    add_samples(high_conf_ham, 35)
    add_samples(high_conf_spam, 35)
    add_samples(borderline, 45)
    add_samples(false_positives, 45)
    add_samples(false_negatives, 45)
    
    print(f"Selected {len(selected_indices)} unique emails for display.")
    
    # Build email items list
    email_list = []
    for idx in sorted(list(selected_indices)):
        prob = float(y_proba[idx])
        raw_text = str(test_raw_texts[idx])
        subject = str(test_subjects[idx])
        
        # Clean up text snippet (remove excessive spaces, newlines)
        clean_snippet = raw_text.replace('\r', '').replace('\n', ' ')
        clean_snippet = ' '.join(clean_snippet.split())[:300] + '...'
        
        # Full text for modal view, escape quotes to prevent JSON parsing issues
        escaped_full_text = raw_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
        escaped_subject = subject.replace('\\', '\\\\').replace('"', '\\"')
        
        email_list.append({
            "id": idx,
            "subject": escaped_subject,
            "snippet": clean_snippet,
            "full_text": escaped_full_text,
            "true_label": str(y_test_arr[idx]),
            "spam_prob": round(prob, 4)
        })
        
    # 6. Generate the HTML Content
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Công cụ Chọn Ngưỡng Quyết Định (Threshold Selector) - vqd</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --success: #10b981;
            --error: #f43f5e;
            --warning: #f59e0b;
            --font-main: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            background-image: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0b0f19 70%);
            color: var(--text-main);
            font-family: var(--font-main);
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.5;
        }}

        header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        header h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        header p {{
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }}

        .badge {{
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #c7d2fe;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        @media (max-width: 1024px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}

        /* Slider section */
        .slider-card {{
            grid-column: 1 / -1;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .slider-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .slider-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #c7d2fe;
        }}

        .slider-val-container {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
            font-family: var(--font-mono);
        }}

        .slider-wrapper {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .range-slider {{
            flex-grow: 1;
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #1f2937;
            outline: none;
            cursor: pointer;
        }}

        .range-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: var(--primary);
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.8);
            border: 2px solid #fff;
            transition: transform 0.1s ease;
        }}

        .range-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.15);
        }}

        .slider-labels {{
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
            font-size: 0.85rem;
            font-family: var(--font-mono);
            padding: 0 0.25rem;
        }}

        /* Metrics grid */
        .metrics-cards-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1rem;
            grid-column: 1 / -1;
        }}

        @media (max-width: 768px) {{
            .metrics-cards-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        .metric-mini-card {{
            background: rgba(31, 41, 55, 0.4);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            transition: all 0.2s ease;
        }}

        .metric-mini-card.active-metric {{
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.15);
        }}

        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }}

        .metric-val {{
            font-size: 1.5rem;
            font-weight: 700;
            font-family: var(--font-mono);
        }}

        .metric-mini-card.fpr-card .metric-val {{
            color: #c084fc;
        }}

        /* Confusion Matrix */
        .matrix-container {{
            display: flex;
            flex-direction: column;
            height: 100%;
        }}

        .matrix-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #c7d2fe;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .matrix-grid {{
            display: grid;
            grid-template-columns: 50px 1fr 1fr;
            grid-template-rows: 35px 1fr 1fr;
            gap: 0.5rem;
            flex-grow: 1;
            text-align: center;
        }}

        .matrix-header-y {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .matrix-header-x {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .matrix-cell {{
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 1rem;
            font-weight: 700;
            transition: all 0.3s ease;
        }}

        .matrix-cell .cell-val {{
            font-size: 1.8rem;
            font-family: var(--font-mono);
        }}

        .matrix-cell .cell-label {{
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }}

        /* Confusion Matrix Styling colors */
        .cell-tn {{
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--success);
        }}

        .cell-tp {{
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #818cf8;
        }}

        .cell-fp {{
            background: rgba(244, 63, 94, 0.12);
            border: 1px solid rgba(244, 63, 94, 0.3);
            color: var(--error);
        }}

        .cell-fn {{
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: var(--warning);
        }}

        .matrix-y-axis-label {{
            grid-row: 2 / span 2;
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
        }}

        .matrix-x-axis-label {{
            grid-column: 2 / span 2;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }}

        /* Canvas Plot */
        .plot-container {{
            display: flex;
            flex-direction: column;
            height: 100%;
        }}

        .plot-canvas-wrapper {{
            position: relative;
            flex-grow: 1;
            min-height: 250px;
            background: rgba(10, 15, 25, 0.5);
            border-radius: 12px;
            border: 1px solid var(--card-border);
            overflow: hidden;
            margin-top: 0.5rem;
        }}

        canvas#plotCanvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        /* Email Inspector section */
        .inspector-card {{
            margin-top: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .inspector-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .inspector-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #c7d2fe;
        }}

        .filter-buttons {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .filter-btn {{
            background: #1f2937;
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }}

        .filter-btn:hover {{
            background: #374151;
            color: var(--text-main);
        }}

        .filter-btn.active-filter {{
            background: var(--primary);
            border-color: var(--primary);
            color: #fff;
        }}

        .search-box {{
            background: #111827;
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-size: 0.85rem;
            outline: none;
            width: 200px;
            font-family: var(--font-main);
        }}

        .search-box:focus {{
            border-color: var(--primary);
        }}

        .email-list {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            max-height: 400px;
            overflow-y: auto;
            padding-right: 0.25rem;
        }}

        /* Custom Scrollbar */
        .email-list::-webkit-scrollbar {{
            width: 6px;
        }}

        .email-list::-webkit-scrollbar-track {{
            background: rgba(0,0,0,0.1);
        }}

        .email-list::-webkit-scrollbar-thumb {{
            background: #374151;
            border-radius: 4px;
        }}

        .email-card {{
            background: rgba(31, 41, 55, 0.3);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            position: relative;
            overflow: hidden;
        }}

        .email-card:hover {{
            background: rgba(55, 65, 81, 0.3);
            border-color: rgba(99, 102, 241, 0.3);
        }}

        .email-meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .email-subject {{
            font-weight: 600;
            font-size: 0.95rem;
            color: #e5e7eb;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 70%;
        }}

        .email-snippet {{
            font-size: 0.85rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .prob-bar-wrapper {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-top: 0.25rem;
        }}

        .prob-bar-bg {{
            flex-grow: 1;
            height: 6px;
            background: #1f2937;
            border-radius: 3px;
            overflow: hidden;
        }}

        .prob-bar-fill {{
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease, background-color 0.3s ease;
        }}

        .prob-val {{
            font-family: var(--font-mono);
            font-size: 0.8rem;
            font-weight: 600;
            min-width: 45px;
            text-align: right;
        }}

        .tag {{
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .tag-true-spam {{
            background: rgba(244, 63, 94, 0.12);
            color: var(--error);
            border: 1px solid rgba(244, 63, 94, 0.2);
        }}

        .tag-true-ham {{
            background: rgba(16, 185, 129, 0.12);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}

        .tag-pred-spam {{
            background: rgba(244, 63, 94, 0.25);
            color: #fda4af;
            border: 1px solid rgba(244, 63, 94, 0.4);
        }}

        .tag-pred-ham {{
            background: rgba(16, 185, 129, 0.25);
            color: #a7f3d0;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}

        .labels-box {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}

        /* Modal styling */
        .modal {{
            display: none;
            position: fixed;
            z-index: 100;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.8);
            backdrop-filter: blur(8px);
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}

        .modal-content {{
            background: #111827;
            border: 1px solid var(--card-border);
            border-radius: 16px;
            max-width: 700px;
            width: 100%;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            animation: modalFadeIn 0.2s ease-out;
        }}

        @keyframes modalFadeIn {{
            from {{ transform: scale(0.95); opacity: 0; }}
            to {{ transform: scale(1); opacity: 1; }}
        }}

        .modal-header {{
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--card-border);
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}

        .modal-title-container {{
            max-width: 85%;
        }}

        .modal-subject {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.25rem;
        }}

        .modal-close {{
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 1.5rem;
            cursor: pointer;
            line-height: 1;
        }}

        .modal-close:hover {{
            color: #fff;
        }}

        .modal-body {{
            padding: 1.5rem;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: var(--font-mono);
            font-size: 0.85rem;
            background: #090d16;
            color: #e5e7eb;
            line-height: 1.6;
        }}

        .modal-footer {{
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--card-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
</head>
<body>

    <header>
        <div>
            <h1>Bộ Chọn Ngưỡng Quyết Định Phân Loại Email (Spam / Ham)</h1>
            <p>Trực quan hóa hiệu năng theo thời gian thực và tương tác lựa chọn ngưỡng phù hợp cho mô hình Naive Bayes cá nhân của vqd</p>
        </div>
        <span class="badge">Naive Bayes Scratch</span>
    </header>

    <div class="dashboard-grid">
        <!-- Slider Control Card -->
        <div class="card slider-card">
            <div class="slider-header">
                <span class="slider-title">Ngưỡng Phân Loại Quyết Định (Threshold)</span>
                <div class="slider-val-container">
                    <span id="thresholdVal">0.50</span>
                </div>
            </div>
            <div class="slider-wrapper">
                <span style="font-family: var(--font-mono); color: var(--text-muted);">0.0</span>
                <input type="range" class="range-slider" id="thresholdSlider" min="0" max="100" value="50">
                <span style="font-family: var(--font-mono); color: var(--text-muted);">1.0</span>
            </div>
            <div class="slider-labels">
                <span>Ưu tiên tìm Spam (FPR tăng, bỏ sót ít)</span>
                <span>Cân bằng (Mặc định)</span>
                <span>Ưu tiên an toàn (Tránh chặn nhầm Ham, FPR → 0)</span>
            </div>
        </div>

        <!-- Metric Mini Cards -->
        <div class="metrics-cards-grid">
            <div class="metric-mini-card" id="card-f1">
                <span class="metric-label">F1-Score</span>
                <span class="metric-val" id="val-f1" style="color: #818cf8;">0.00%</span>
            </div>
            <div class="metric-mini-card" id="card-precision">
                <span class="metric-label">Precision</span>
                <span class="metric-val" id="val-precision" style="color: #fb7185;">0.00%</span>
            </div>
            <div class="metric-mini-card" id="card-recall">
                <span class="metric-label">Recall</span>
                <span class="metric-val" id="val-recall" style="color: #34d399;">0.00%</span>
            </div>
            <div class="metric-mini-card" id="card-accuracy">
                <span class="metric-label">Accuracy</span>
                <span class="metric-val" id="val-accuracy" style="color: #fbbf24;">0.00%</span>
            </div>
            <div class="metric-mini-card fpr-card" id="card-fpr">
                <span class="metric-label">FPR (Chặn nhầm)</span>
                <span class="metric-val" id="val-fpr">0.00%</span>
            </div>
        </div>

        <!-- Confusion Matrix Card -->
        <div class="card">
            <div class="matrix-container">
                <div class="matrix-title">
                    <span>Ma Trận Nhầm Lẫn (Confusion Matrix)</span>
                    <span style="font-size: 0.85rem; font-weight: normal; color: var(--text-muted);" id="matrixTotalLabel"></span>
                </div>
                <div class="matrix-grid">
                    <div></div>
                    <div class="matrix-header-x">Dự đoán HAM</div>
                    <div class="matrix-header-x">Dự đoán SPAM</div>
                    
                    <div class="matrix-header-y">Thực tế HAM</div>
                    <div class="matrix-cell cell-tn" id="cell-tn">
                        <span class="cell-val" id="val-tn">0</span>
                        <span class="cell-label">True Negative (TN)</span>
                    </div>
                    <div class="matrix-cell cell-fp" id="cell-fp">
                        <span class="cell-val" id="val-fp">0</span>
                        <span class="cell-label">False Positive (FP)</span>
                    </div>
                    
                    <div class="matrix-header-y">Thực tế SPAM</div>
                    <div class="matrix-cell cell-fn" id="cell-fn">
                        <span class="cell-val" id="val-fn">0</span>
                        <span class="cell-label">False Negative (FN)</span>
                    </div>
                    <div class="matrix-cell cell-tp" id="cell-tp">
                        <span class="cell-val" id="val-tp">0</span>
                        <span class="cell-label">True Positive (TP)</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Line Plot Card -->
        <div class="card">
            <div class="plot-container">
                <div style="font-size: 1.1rem; font-weight: 600; color: #c7d2fe; display: flex; justify-content: space-between; align-items: center;">
                    <span>Biểu Đồ Đường Các Độ Đo Theo Ngưỡng</span>
                    <div style="display: flex; gap: 0.75rem; font-size: 0.75rem;">
                        <span style="color: #818cf8;">● F1</span>
                        <span style="color: #fb7185;">● Prec</span>
                        <span style="color: #34d399;">● Recall</span>
                        <span style="color: #c084fc;">● FPR</span>
                    </div>
                </div>
                <div class="plot-canvas-wrapper">
                    <canvas id="plotCanvas"></canvas>
                </div>
            </div>
        </div>
    </div>

    <!-- Email Inspector Card -->
    <div class="card inspector-card">
        <div class="inspector-header">
            <span class="inspector-title">Trình Kiểm Tra Email Mẫu (Email Inspector)</span>
            <div class="filter-buttons">
                <button class="filter-btn active-filter" data-filter="all">Tất cả mẫu ({len(email_list)})</button>
                <button class="filter-btn" data-filter="borderline">Ranh giới (0.40 - 0.60)</button>
                <button class="filter-btn" data-filter="false_positive">Báo động giả (FP)</button>
                <button class="filter-btn" data-filter="false_negative">Bỏ sót (FN)</button>
                <button class="filter-btn" data-filter="true_spam">Spam đúng</button>
                <button class="filter-btn" data-filter="true_ham">Ham đúng</button>
            </div>
            <input type="text" class="search-box" id="searchBox" placeholder="Tìm kiếm trong email...">
        </div>

        <div class="email-list" id="emailList">
            <!-- Dynamic elements will be inserted here -->
        </div>
    </div>

    <!-- Details Modal -->
    <div class="modal" id="emailModal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title-container">
                    <div class="modal-subject" id="modalSubject">Subject Line</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); display: flex; gap: 0.75rem; align-items: center; margin-top: 0.25rem;">
                        <span>ID Mẫu: <b id="modalId">0</b></span>
                        <span>Nhãn thực tế: <span class="tag" id="modalTrueTag">HAM</span></span>
                        <span>Xác suất Spam: <b id="modalProb" style="color: var(--primary);">0.00%</b></span>
                    </div>
                </div>
                <button class="modal-close" id="modalClose">&times;</button>
            </div>
            <div class="modal-body" id="modalBody">
                Email content text...
            </div>
            <div class="modal-footer">
                <div class="labels-box">
                    <span style="font-size: 0.85rem; color: var(--text-muted);">Dự đoán với ngưỡng hiện tại:</span>
                    <span class="tag" id="modalPredTag">HAM</span>
                </div>
                <button class="filter-btn active-filter" id="modalCloseBtn" style="padding: 0.4rem 1.2rem;">Đóng</button>
            </div>
        </div>
    </div>

    <script>
        // Data generated from Python
        const thresholdsData = {json.dumps(metrics_data, indent=2)};
        const emailsData = {json.dumps(email_list, indent=2)};

        // State variables
        let currentThreshold = 0.50;
        let activeFilter = 'all';
        let searchQuery = '';

        // DOM Elements
        const slider = document.getElementById('thresholdSlider');
        const thresholdVal = document.getElementById('thresholdVal');
        
        // Metrics
        const valF1 = document.getElementById('val-f1');
        const valPrecision = document.getElementById('val-precision');
        const valRecall = document.getElementById('val-recall');
        const valAccuracy = document.getElementById('val-accuracy');
        const valFpr = document.getElementById('val-fpr');
        
        // Confusion Matrix cells
        const valTn = document.getElementById('val-tn');
        const valFp = document.getElementById('val-fp');
        const valFn = document.getElementById('val-fn');
        const valTp = document.getElementById('val-tp');
        const matrixTotalLabel = document.getElementById('matrixTotalLabel');
        
        // Plot canvas
        const canvas = document.getElementById('plotCanvas');
        const ctx = canvas.getContext('2d');
        
        // Email Inspector elements
        const emailList = document.getElementById('emailList');
        const searchBox = document.getElementById('searchBox');
        const filterBtns = document.querySelectorAll('.filter-btn');
        
        // Modal elements
        const modal = document.getElementById('emailModal');
        const modalSubject = document.getElementById('modalSubject');
        const modalId = document.getElementById('modalId');
        const modalTrueTag = document.getElementById('modalTrueTag');
        const modalProb = document.getElementById('modalProb');
        const modalBody = document.getElementById('modalBody');
        const modalPredTag = document.getElementById('modalPredTag');
        const modalClose = document.getElementById('modalClose');
        const modalCloseBtn = document.getElementById('modalCloseBtn');

        // Init App
        function init() {{
            setupCanvas();
            updateUI();
            
            // Event Listeners
            slider.addEventListener('input', function() {{
                currentThreshold = parseFloat(this.value) / 100;
                thresholdVal.textContent = currentThreshold.toFixed(2);
                updateUI();
            }});

            searchBox.addEventListener('input', function() {{
                searchQuery = this.value.toLowerCase();
                renderEmailList();
            }});

            filterBtns.forEach(btn => {{
                btn.addEventListener('click', function() {{
                    if (this.id === 'modalCloseBtn') return; // Ignore modal close
                    filterBtns.forEach(b => b.classList.remove('active-filter'));
                    this.classList.add('active-filter');
                    activeFilter = this.getAttribute('data-filter');
                    renderEmailList();
                }});
            }});

            // Modal listeners
            modalClose.addEventListener('click', closeModal);
            modalCloseBtn.addEventListener('click', closeModal);
            window.addEventListener('click', function(e) {{
                if (e.target === modal) closeModal();
            }});
            
            window.addEventListener('resize', () => {{
                setupCanvas();
                drawPlot();
            }});
        }}

        // Handle canvas sizing and DPI scaling
        function setupCanvas() {{
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
        }}

        // Update UI components
        function updateUI() {{
            // Find current stats
            const stats = thresholdsData.find(d => Math.abs(d.threshold - currentThreshold) < 0.005) || thresholdsData[50];
            
            // Update Mini stats
            valF1.textContent = stats.f1.toFixed(2) + '%';
            valPrecision.textContent = stats.precision.toFixed(2) + '%';
            valRecall.textContent = stats.recall.toFixed(2) + '%';
            valAccuracy.textContent = stats.accuracy.toFixed(2) + '%';
            valFpr.textContent = stats.fpr.toFixed(2) + '%';
            
            // Highlight F1 if it's highest or high
            
            // Update Confusion Matrix
            valTn.textContent = stats.tn.toLocaleString();
            valFp.textContent = stats.fp.toLocaleString();
            valFn.textContent = stats.fn.toLocaleString();
            valTp.textContent = stats.tp.toLocaleString();
            
            const total = stats.tn + stats.fp + stats.fn + stats.tp;
            matrixTotalLabel.textContent = `Tập Test: ${{total.toLocaleString()}} mẫu`;
            
            // Draw chart
            drawPlot();
            
            // Render Email Inspector list
            renderEmailList();
        }}

        // Custom Javascript Plot drawing
        function drawPlot() {{
            const w = canvas.width / (window.devicePixelRatio || 1);
            const h = canvas.height / (window.devicePixelRatio || 1);
            ctx.clearRect(0, 0, w, h);
            
            const paddingLeft = 40;
            const paddingRight = 15;
            const paddingTop = 15;
            const paddingBottom = 30;
            
            const graphWidth = w - paddingLeft - paddingRight;
            const graphHeight = h - paddingTop - paddingBottom;
            
            // Draw Grid Lines
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {{
                const y = paddingTop + (graphHeight * i / 4);
                ctx.beginPath();
                ctx.moveTo(paddingLeft, y);
                ctx.lineTo(w - paddingRight, y);
                ctx.stroke();
                
                // Y-labels (percentage)
                ctx.fillStyle = '#9ca3af';
                ctx.font = '10px var(--font-mono)';
                ctx.textAlign = 'right';
                ctx.fillText((100 - (25 * i)) + '%', paddingLeft - 8, y + 3);
            }}
            
            for (let i = 0; i <= 5; i++) {{
                const x = paddingLeft + (graphWidth * i / 5);
                ctx.beginPath();
                ctx.moveTo(x, paddingTop);
                ctx.lineTo(x, h - paddingBottom);
                ctx.stroke();
                
                // X-labels (threshold)
                ctx.fillStyle = '#9ca3af';
                ctx.font = '10px var(--font-mono)';
                ctx.textAlign = 'center';
                ctx.fillText((i / 5).toFixed(1), x, h - paddingBottom + 16);
            }}
            
            // Draw Metric lines: F1, Precision, Recall, FPR
            const drawMetricLine = (dataKey, color) => {{
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                
                thresholdsData.forEach((d, idx) => {{
                    const x = paddingLeft + (graphWidth * d.threshold);
                    const val = d[dataKey] / 100; // normalized to 0-1
                    const y = paddingTop + graphHeight * (1 - val);
                    
                    if (idx === 0) {{
                        ctx.moveTo(x, y);
                    }} else {{
                        ctx.lineTo(x, y);
                    }}
                }});
                ctx.stroke();
            }};
            
            // Draw Lines
            drawMetricLine('fpr', '#c084fc');
            drawMetricLine('recall', '#34d399');
            drawMetricLine('precision', '#fb7185');
            drawMetricLine('f1', '#818cf8');
            
            // Draw Current Threshold vertical cursor
            const cursorX = paddingLeft + (graphWidth * currentThreshold);
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.7)';
            ctx.setLineDash([5, 5]);
            ctx.lineWidth = 1.5;
            ctx.moveTo(cursorX, paddingTop);
            ctx.lineTo(cursorX, h - paddingBottom);
            ctx.stroke();
            ctx.setLineDash([]); // reset dash
            
            // Draw cursor circle marker at intersection with F1
            const currentStats = thresholdsData.find(d => Math.abs(d.threshold - currentThreshold) < 0.005) || thresholdsData[50];
            const f1Val = currentStats.f1 / 100;
            const cursorY = paddingTop + graphHeight * (1 - f1Val);
            
            ctx.beginPath();
            ctx.fillStyle = '#fff';
            ctx.strokeStyle = '#6366f1';
            ctx.lineWidth = 2.5;
            ctx.arc(cursorX, cursorY, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }}

        // Render Email Inspector list based on filters
        function renderEmailList() {{
            emailList.innerHTML = '';
            
            const filteredEmails = emailsData.filter(email => {{
                // Search filter
                const matchesSearch = email.subject.toLowerCase().includes(searchQuery) || 
                                      email.snippet.toLowerCase().includes(searchQuery);
                if (!matchesSearch) return false;
                
                // Prediction status based on currentThreshold
                const isPredictedSpam = email.spam_prob >= currentThreshold;
                const predLabel = isPredictedSpam ? 'spam' : 'ham';
                
                // Category filters
                if (activeFilter === 'all') return true;
                if (activeFilter === 'borderline') {{
                    return email.spam_prob >= 0.40 && email.spam_prob <= 0.60;
                }}
                if (activeFilter === 'false_positive') {{
                    return email.true_label === 'ham' && predLabel === 'spam';
                }}
                if (activeFilter === 'false_negative') {{
                    return email.true_label === 'spam' && predLabel === 'ham';
                }}
                if (activeFilter === 'true_spam') {{
                    return email.true_label === 'spam' && predLabel === 'spam';
                }}
                if (activeFilter === 'true_ham') {{
                    return email.true_label === 'ham' && predLabel === 'ham';
                }}
                return true;
            }});

            if (filteredEmails.length === 0) {{
                emailList.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 2rem; font-size: 0.9rem;">Không tìm thấy mẫu email phù hợp với bộ lọc hiện tại.</div>';
                return;
            }}

            filteredEmails.forEach(email => {{
                const isPredictedSpam = email.spam_prob >= currentThreshold;
                const predLabel = isPredictedSpam ? 'spam' : 'ham';
                const isError = email.true_label !== predLabel;
                
                // Custom probability bar color based on prediction
                let barColor = 'var(--success)'; // ham
                if (email.spam_prob > 0.8) barColor = 'var(--error)';
                else if (email.spam_prob >= 0.4) barColor = 'var(--warning)';

                const card = document.createElement('div');
                card.className = 'email-card';
                if (isError) {{
                    card.style.borderColor = 'rgba(244, 63, 94, 0.2)';
                    card.style.background = 'rgba(244, 63, 94, 0.03)';
                }}
                
                card.innerHTML = `
                    <div class="email-meta">
                        <span class="email-subject">${{escapeHtml(email.subject)}}</span>
                        <div class="labels-box">
                            <span class="tag tag-true-${{email.true_label}}">Thực tế: ${{email.true_label}}</span>
                            <span class="tag tag-pred-${{predLabel}}">Dự đoán: ${{predLabel}}</span>
                        </div>
                    </div>
                    <div class="email-snippet">${{escapeHtml(email.snippet)}}</div>
                    <div class="prob-bar-wrapper">
                        <span style="font-size: 0.75rem; color: var(--text-muted);">Khả năng Spam:</span>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: ${{email.spam_prob * 100}}%; background-color: ${{barColor}};"></div>
                        </div>
                        <span class="prob-val" style="color: ${{barColor}}">${{(email.spam_prob * 100).toFixed(1)}}%</span>
                    </div>
                `;
                
                card.addEventListener('click', () => showEmailModal(email));
                emailList.appendChild(card);
            }});
        }}

        // Helper to escape HTML to prevent page breaks
        function escapeHtml(text) {{
            const map = {{
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            }};
            return text.replace(/[&<>"']/g, function(m) {{ return map[m]; }});
        }}

        // Show Email Details Modal
        function showEmailModal(email) {{
            const isPredictedSpam = email.spam_prob >= currentThreshold;
            const predLabel = isPredictedSpam ? 'spam' : 'ham';
            
            modalSubject.textContent = email.subject || '(No Subject)';
            modalId.textContent = email.id;
            modalProb.textContent = (email.spam_prob * 100).toFixed(2) + '%';
            
            // True Tag
            modalTrueTag.className = `tag tag-true-${{email.true_label}}`;
            modalTrueTag.textContent = email.true_label.toUpperCase();
            
            // Pred Tag
            modalPredTag.className = `tag tag-pred-${{predLabel}}`;
            modalPredTag.textContent = predLabel.toUpperCase();
            
            // Body text
            modalBody.textContent = email.full_text;
            
            modal.style.display = 'flex';
        }}

        function closeModal() {{
            modal.style.display = 'none';
        }}

        // Run application
        window.onload = init;
    </script>
</body>
</html>
"""
    
    # Save the HTML output
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Successfully generated visual dashboard HTML at: {output_html_path}")

if __name__ == "__main__":
    generate_dashboard()
