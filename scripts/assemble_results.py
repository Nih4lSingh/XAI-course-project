import shutil
import json
from pathlib import Path
import pandas as pd

# 1. Comparison
comp_src = Path('results/paper_vs_reproduction.csv')
comp_dst = Path('results/comparison/paper_vs_replication.csv')
comp_dst.parent.mkdir(parents=True, exist_ok=True)
if comp_src.exists():
    shutil.copy(comp_src, comp_dst)

# 2. Confusion matrices & curves
models = {
    'dnn': ('nsl_selected_dnn', 'unsw_selected_dnn'),
    '1d_cnn': ('nsl_selected_1dcnn', 'unsw_selected_1dcnn'),
    '2d_cnn': ('nsl_selected_2dcnn', 'unsw_selected_2dcnn'),
}

for m_key, (nsl_id, unsw_id) in models.items():
    m_dir = Path(f'results/{m_key}')
    m_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy CM
    nsl_cm = Path(f'results/confusion_matrices/{nsl_id}_cm.png')
    unsw_cm = Path(f'results/confusion_matrices/{unsw_id}_cm.png')
    if nsl_cm.exists():
        shutil.copy(nsl_cm, m_dir / 'confusion_matrix_nsl.png')
    if unsw_cm.exists():
        shutil.copy(unsw_cm, m_dir / 'confusion_matrix_unsw.png')
        
    # Copy Curves
    nsl_curve = Path(f'results/training_curves/{nsl_id}_accuracy.png')
    unsw_curve = Path(f'results/training_curves/{unsw_id}_accuracy.png')
    if nsl_curve.exists():
        shutil.copy(nsl_curve, m_dir / 'training_curves_nsl.png')
    if unsw_curve.exists():
        shutil.copy(unsw_curve, m_dir / 'training_curves_unsw.png')
        
    # Generate metrics.csv
    nsl_json = json.load(open(f'results/canonical/{nsl_id}/metrics.json'))
    unsw_json = json.load(open(f'results/canonical/{unsw_id}/metrics.json'))
    
    label = m_key.upper().replace('_', '-')
    rows = [
        {
            'Dataset': f'NSL-KDD {label}',
            'Accuracy': round(nsl_json['accuracy'], 6),
            'Precision Macro': round(nsl_json['precision_macro'], 6),
            'Precision Weighted': round(nsl_json['precision_weighted'], 6),
            'Recall Macro': round(nsl_json['recall_macro'], 6),
            'Recall Weighted': round(nsl_json['recall_weighted'], 6),
            'F1 Macro': round(nsl_json['f1_macro'], 6),
            'F1 Weighted': round(nsl_json['f1_weighted'], 6)
        },
        {
            'Dataset': f'UNSW-NB15 {label}',
            'Accuracy': round(unsw_json['accuracy'], 6),
            'Precision Macro': round(unsw_json['precision_macro'], 6),
            'Precision Weighted': round(unsw_json['precision_weighted'], 6),
            'Recall Macro': round(unsw_json['recall_macro'], 6),
            'Recall Weighted': round(unsw_json['recall_weighted'], 6),
            'F1 Macro': round(unsw_json['f1_macro'], 6),
            'F1 Weighted': round(unsw_json['f1_weighted'], 6)
        }
    ]
    pd.DataFrame(rows).to_csv(m_dir / 'metrics.csv', index=False)
    print(f'Successfully assembled results for {m_key}')

print('All results assembled.')
