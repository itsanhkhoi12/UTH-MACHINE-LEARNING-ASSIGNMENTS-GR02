from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cloudpickle
import pandas as pd


def _json_safe(value: Any):
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_model_package(model, model_name, best_params, metrics, save_dir='./models', extra=None):
    """Lưu mô hình thật bằng cloudpickle và lưu metrics ở file JSON riêng."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / f'{model_name}.pkl'
    with model_path.open('wb') as f:
        cloudpickle.dump(model, f)

    metadata = {
        'model_name': model_name,
        'best_params': _json_safe(dict(best_params or {})),
        'metrics': _json_safe(dict(metrics or {})),
        'extra': _json_safe(dict(extra or {})),
        'model_file': model_path.name,
    }
    metadata_path = save_path / f'{model_name}_metrics.json'
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    print(f'Đã lưu mô hình: {model_path}')
    print(f'Đã lưu kết quả: {metadata_path}')
    return str(model_path)


def load_saved_model(model_path):
    """Nạp lại mô hình đã lưu mà không cần import class từ notebook."""
    with Path(model_path).open('rb') as f:
        return cloudpickle.load(f)


def generate_report(model_dir='./models'):
    """Tạo bảng tổng hợp từ các file *_metrics.json."""
    rows = []
    for path in sorted(Path(model_dir).glob('*_metrics.json')):
        metadata = json.loads(path.read_text(encoding='utf-8'))
        row = {'Mô hình': metadata.get('model_name', path.stem)}
        row.update(metadata.get('metrics', {}))
        params = metadata.get('best_params', {})
        row['Tham số tốt nhất'] = ', '.join(f'{k}={v}' for k, v in params.items())
        row['Target'] = metadata.get('extra', {}).get('target', '')
        row['File mô hình'] = metadata.get('model_file', '')
        rows.append(row)

    df = pd.DataFrame(rows)
    order = [
        'linear_regression_scratch',
        'decision_tree_scratch',
        'random_forest_scratch',
        'model_lightgbm_scratch',
    ]
    if not df.empty:
        df['_order'] = df['Mô hình'].map({name: i for i, name in enumerate(order)}).fillna(999)
        df = df.sort_values('_order').drop(columns='_order').reset_index(drop=True)
    return df
