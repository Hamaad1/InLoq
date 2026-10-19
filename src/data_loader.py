"""
Data loading and preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load CSV, remove BT outliers with One-Class SVM, add Bxy column."""
    df = pd.read_csv(csv_path)
    oc = OneClassSVM(kernel='rbf', nu=0.05)
    oc.fit(df[['BT']])
    mask = oc.predict(df[['BT']]) == 1
    clean = df[mask].copy()
    if 'Bxy' not in clean.columns:
        clean['Bxy'] = np.sqrt(clean['Bx']**2 + clean['By']**2)
    print(f'✅ Clean dataset shape: {clean.shape}')
    print(f'✅ Columns: {clean.columns.tolist()}')
    return clean


def prepare_data(features, df, y, random_state: int = 42):
    """80/10/10 split + StandardScaler. Returns float32 arrays."""
    X = df[features].values.astype(np.float32)
    yarr = (y.values if hasattr(y, 'values') else y).astype(np.float32)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, yarr, test_size=0.2, random_state=random_state)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=random_state)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_va = sc.transform(X_va).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)

    return X_tr, X_va, X_te, y_tr, y_va, y_te, sc



def prepare_sequences(X, y, seq_len: int = 1):
    """
    Reshape (N, F) → (N', seq_len, F) sliding windows.
    seq_len == 1  =>  simple unsqueeze on axis-1.
    """
    Xv = X.values if hasattr(X, 'values') else X
    yv = y.values if hasattr(y, 'values') else y

    if seq_len <= 1:
        return Xv.reshape(Xv.shape[0], 1, Xv.shape[1]), yv

    shape   = (Xv.shape[0] - seq_len + 1, seq_len, Xv.shape[1])
    strides = (Xv.strides[0], Xv.strides[0], Xv.strides[1])
    Xseq    = np.lib.stride_tricks.as_strided(Xv, shape=shape, strides=strides).copy()
    return Xseq, yv[seq_len - 1:]
