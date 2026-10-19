# src/parameter_counter.py
"""
Parameter counting utilities for all models.
Used for complexity analysis and generating parameter comparison tables/plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Constants
SEQUENCE_LENGTH = 6
INPUT_DIMS = [1, 2, 3, 4]

MODEL_PALETTE = {
    'InLoq (total)'   : '#1f77b4',
    'InLoq (quantum)' : '#aec7e8',
    'Classical ANN'   : '#ff7f0e',
    'LSTM'            : '#2ca02c',
    'GRU'             : '#d62728',
    'BiLSTM'          : '#9467bd',
}

PLOT_PALETTE = {
    'InLoq (Proposed)'  : '#1f77b4',
    'Classical ANN'     : '#ff7f0e',
    'LSTM'              : '#2ca02c',
    'GRU'               : '#d62728',
    'BiLSTM'            : '#9467bd',
    'InLoq (quantum only)': '#aec7e8',
}


def _dense(in_dim: int, out_dim: int) -> int:
    """Calculate parameters for a dense layer: (in * out) + out"""
    return in_dim * out_dim + out_dim


def params_InLoq(input_dim: int) -> dict:
    """
    Calculate InLoQ parameter count.
    
    Architecture:
    - Dense(n_q, tanh) — classical pre-processing
    - QuantumLayer(2, n_q, 3) — 2 × n_q × 3 trainable rotation angles
    - Dense(16, relu) — post-quantum 1
    - Dense(8, relu) — post-quantum 2
    - Dense(2, linear) — output
    """
    n_q = min(max(input_dim, 2), 8)

    classical_pre = _dense(input_dim, n_q)
    quantum = 2 * n_q * 3
    post_q1 = _dense(n_q, 16)
    post_q2 = _dense(16, 8)
    output = _dense(8, 2)

    classical_only = classical_pre + post_q1 + post_q2 + output
    total = classical_only + quantum

    return {
        'input_dim': input_dim,
        'n_qubits': n_q,
        'classical_pre': classical_pre,
        'quantum': quantum,
        'post_q1': post_q1,
        'post_q2': post_q2,
        'output': output,
        'classical_only': classical_only,
        'total': total,
    }


def get_matched_ann_width(input_dim: int, target_params: int) -> int:
    """Find ANN width that matches target parameter count."""
    for w in range(8, 512, 2):
        test_model = Sequential([
            Dense(w, input_dim=input_dim, activation='relu'),
            Dense(w // 2, activation='relu'),
            Dense(2, activation='linear')
        ])
        if test_model.count_params() >= target_params:
            return w
    return 32


def params_ann(input_dim: int, target_params: int = None) -> dict:
    """
    Calculate Classical ANN parameter count.
    If target_params provided, find width that matches InLoQ.
    """
    if target_params is not None:
        width = get_matched_ann_width(input_dim, target_params)
    else:
        width = 32
    
    l1 = _dense(input_dim, width)
    l2 = _dense(width, width // 2)
    output = _dense(width // 2, 2)
    total = l1 + l2 + output
    
    return {
        'input_dim': input_dim,
        'width': width,
        'layer1': l1,
        'layer2': l2,
        'output': output,
        'total': total,
    }


def _lstm_cell_params(input_dim: int, units: int) -> int:
    """Standard LSTM cell parameter calculation."""
    return 4 * ((input_dim + units) * units + units)


def _gru_cell_params(input_dim: int, units: int) -> int:
    """
    Keras GRU parameter calculation.
    Includes kernel, recurrent, and bias parameters.
    """
    kernel = input_dim * units * 3
    recurrent = units * units * 3
    bias = 2 * units * 3
    return kernel + recurrent + bias


def params_lstm(input_dim: int, seq_len: int = SEQUENCE_LENGTH) -> dict:
    """
    Calculate LSTM parameter count.
    
    Architecture:
    - LSTM(16, return_sequences=False)
    - Dropout(0.1) — no parameters
    - Dense(8, relu)
    - Dense(2, linear)
    """
    lstm = _lstm_cell_params(input_dim, 16)
    dense1 = _dense(16, 8)
    output = _dense(8, 2)
    total = lstm + dense1 + output

    return {
        'input_dim': input_dim,
        'sequence_length': seq_len,
        'lstm_16': lstm,
        'dense_8': dense1,
        'output': output,
        'total': total,
    }


def params_gru(input_dim: int, seq_len: int = SEQUENCE_LENGTH) -> dict:
    """
    Calculate GRU parameter count.
    
    Architecture:
    - GRU(16, return_sequences=False)
    - Dropout(0.1) — no parameters
    - Dense(8, relu)
    - Dense(2, linear)
    """
    gru = _gru_cell_params(input_dim, 16)
    dense1 = _dense(16, 8)
    output = _dense(8, 2)
    total = gru + dense1 + output

    return {
        'input_dim': input_dim,
        'sequence_length': seq_len,
        'gru_16': gru,
        'dense_8': dense1,
        'output': output,
        'total': total,
    }


def params_bilstm(input_dim: int, seq_len: int = SEQUENCE_LENGTH) -> dict:
    """
    Calculate BiLSTM parameter count.
    
    Architecture:
    - Bidirectional(LSTM(8)) → forward + backward LSTM(8)
    - Dropout(0.1) — no parameters
    - Dense(8, relu)
    - Dense(2, linear)
    """
    forward = _lstm_cell_params(input_dim, 8)
    backward = _lstm_cell_params(input_dim, 8)
    bilstm = forward + backward
    dense1 = _dense(16, 8)  # merged output dim = 8 + 8 = 16
    output = _dense(8, 2)
    total = bilstm + dense1 + output

    return {
        'input_dim': input_dim,
        'sequence_length': seq_len,
        'bilstm_fwd_8': forward,
        'bilstm_bwd_8': backward,
        'bilstm_total': bilstm,
        'dense_8': dense1,
        'output': output,
        'total': total,
    }


def build_summary(input_dims=INPUT_DIMS):
    """
    Build summary DataFrames for parameter comparison.
    
    Returns:
        df_wide: One row per input_dim, columns per model
        df_long: Tidy format for seaborn plots
    """
    wide_rows = []
    long_rows = []

    for d in input_dims:
        rq = params_InLoq(d)
        target = rq['total']
        an = params_ann(d, target_params=target)
        ls = params_lstm(d)
        gr = params_gru(d)
        bl = params_bilstm(d)

        wide_rows.append({
            'Input_Dim': d,
            'InLoq_Quantum_Only': rq['quantum'],
            'InLoq_Classical_Only': rq['classical_only'],
            'InLoq_Total': rq['total'],
            'ANN_Total': an['total'],
            'LSTM_Total': ls['total'],
            'GRU_Total': gr['total'],
            'BiLSTM_Total': bl['total'],
        })

        for model_name, total in [
            ('InLoq (Proposed)', rq['total']),
            ('Classical ANN', an['total']),
            ('LSTM', ls['total']),
            ('GRU', gr['total']),
            ('BiLSTM', bl['total']),
        ]:
            long_rows.append({
                'Input Dimension': d,
                'Model': model_name,
                'Parameters': total,
            })

        # Extra row for quantum portion only
        long_rows.append({
            'Input Dimension': d,
            'Model': 'InLoq (quantum only)',
            'Parameters': rq['quantum'],
        })

    df_wide = pd.DataFrame(wide_rows)
    df_long = pd.DataFrame(long_rows)
    return df_wide, df_long


# ============================================================
# ADD THIS SECTION AT THE END OF THE FILE
# ============================================================

# Pre-compute InLoq parameters for all input dimensions
params_inloq = {}
params_quantum_only = {}

for dim in INPUT_DIMS:
    rq = params_InLoq(dim)
    params_inloq[dim] = rq['total']
    params_quantum_only[dim] = rq['quantum']

# Also create a breakdown dictionary for detailed plotting
params_inloq_breakdown = {
    dim: {
        'classical_pre': params_InLoq(dim)['classical_pre'],
        'quantum': params_InLoq(dim)['quantum'],
        'post_q1': params_InLoq(dim)['post_q1'],
        'post_q2': params_InLoq(dim)['post_q2'],
        'output': params_InLoq(dim)['output'],
        'total': params_InLoq(dim)['total']
    }
    for dim in INPUT_DIMS
}

# # src/parameter_counter.py
# """
# Parameter counting utilities for all models.
# Used for complexity analysis and generating parameter comparison tables/plots.
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # Constants
# SEQUENCE_LENGTH = 6
# INPUT_DIMS = [1, 2, 3, 4]

# MODEL_PALETTE = {
#     'InLoq (total)'   : '#1f77b4',
#     'InLoq (quantum)' : '#aec7e8',
#     'Classical ANN'   : '#ff7f0e',
#     'LSTM'            : '#2ca02c',
#     'GRU'             : '#d62728',
#     'BiLSTM'          : '#9467bd',
# }

# PLOT_PALETTE = {
#     'InLoq (Proposed)'  : '#1f77b4',
#     'Classical ANN'     : '#ff7f0e',
#     'LSTM'              : '#2ca02c',
#     'GRU'               : '#d62728',
#     'BiLSTM'            : '#9467bd',
#     'InLoq (quantum only)': '#aec7e8',
# }


# def _dense(in_dim: int, out_dim: int) -> int:
#     """Calculate parameters for a dense layer: (in * out) + out"""
#     return in_dim * out_dim + out_dim


# def params_InLoq(input_dim: int) -> dict:
#     """
#     Calculate InLoQ parameter count.
    
#     Architecture:
#     - Dense(n_q, tanh) — classical pre-processing
#     - QuantumLayer(2, n_q, 3) — 2 × n_q × 3 trainable rotation angles
#     - Dense(16, relu) — post-quantum 1
#     - Dense(8, relu) — post-quantum 2
#     - Dense(2, linear) — output
#     """
#     n_q = min(max(input_dim, 2), 8)

#     classical_pre = _dense(input_dim, n_q)
#     quantum = 2 * n_q * 3
#     post_q1 = _dense(n_q, 16)
#     post_q2 = _dense(16, 8)
#     output = _dense(8, 2)

#     classical_only = classical_pre + post_q1 + post_q2 + output
#     total = classical_only + quantum

#     return {
#         'input_dim': input_dim,
#         'n_qubits': n_q,
#         'classical_pre': classical_pre,
#         'quantum': quantum,
#         'post_q1': post_q1,
#         'post_q2': post_q2,
#         'output': output,
#         'classical_only': classical_only,
#         'total': total,
#     }


# def get_matched_ann_width(input_dim: int, target_params: int) -> int:
#     """Find ANN width that matches target parameter count."""
#     for w in range(8, 512, 2):
#         test_model = Sequential([
#             Dense(w, input_dim=input_dim, activation='relu'),
#             Dense(w // 2, activation='relu'),
#             Dense(2, activation='linear')
#         ])
#         if test_model.count_params() >= target_params:
#             return w
#     return 32


# def params_ann(input_dim: int, target_params: int = None) -> dict:
#     """
#     Calculate Classical ANN parameter count.
#     If target_params provided, find width that matches InLoQ.
#     """
#     if target_params is not None:
#         width = get_matched_ann_width(input_dim, target_params)
#     else:
#         width = 32
    
#     l1 = _dense(input_dim, width)
#     l2 = _dense(width, width // 2)
#     output = _dense(width // 2, 2)
#     total = l1 + l2 + output
    
#     return {
#         'input_dim': input_dim,
#         'width': width,
#         'layer1': l1,
#         'layer2': l2,
#         'output': output,
#         'total': total,
#     }


# def _lstm_cell_params(input_dim: int, units: int) -> int:
#     """Standard LSTM cell parameter calculation."""
#     return 4 * ((input_dim + units) * units + units)


# def _gru_cell_params(input_dim: int, units: int) -> int:
#     """
#     Keras GRU parameter calculation.
#     Includes kernel, recurrent, and bias parameters.
#     """
#     kernel = input_dim * units * 3
#     recurrent = units * units * 3
#     bias = 2 * units * 3
#     return kernel + recurrent + bias


# def params_lstm(input_dim: int, seq_len: int = SEQUENCE_LENGTH) -> dict:
#     """
#     Calculate LSTM parameter count.
    
#     Architecture:
#     - LSTM(16, return_sequences=False)
#     - Dropout(0.1) — no parameters
#     - Dense(8, relu)
#     - Dense(2, linear)
#     """
#     lstm = _lstm_cell_params(input_dim, 16)
#     dense1 = _dense(16, 8)
#     output = _dense(8, 2)
#     total = lstm + dense1 + output

#     return {
#         'input_dim': input_dim,
#         'sequence_length': seq_len,
#         'lstm_16': lstm,
#         'dense_8': dense1,
#         'output': output,
#         'total': total,
#     }


# def params_gru(input_dim: int, seq_len: int = SEQUENCE_LENGTH) -> dict:
#     """
#     Calculate GRU parameter count.
    
#     Architecture:
#     - GRU(16, return_sequences=False)
#     - Dropout(0.1) — no parameters
#     - Dense(8, relu)
#     - Dense(2, linear)
#     """
#     gru = _gru_cell_params(input_dim, 16)
#     dense1 = _dense(16, 8)
#     output = _dense(8, 2)
#     total = gru + dense1 + output

#     return {
#         'input_dim': input_dim,
#         'sequence_length': seq_len,
#         'gru_16': gru,
#         'dense_8': dense1,
#         'output': output,
#         'total': total,
#     }


# def params_bilstm(input_dim: int, seq_len: int = SEQUENCE_LENGTH) -> dict:
#     """
#     Calculate BiLSTM parameter count.
    
#     Architecture:
#     - Bidirectional(LSTM(8)) → forward + backward LSTM(8)
#     - Dropout(0.1) — no parameters
#     - Dense(8, relu)
#     - Dense(2, linear)
#     """
#     forward = _lstm_cell_params(input_dim, 8)
#     backward = _lstm_cell_params(input_dim, 8)
#     bilstm = forward + backward
#     dense1 = _dense(16, 8)  # merged output dim = 8 + 8 = 16
#     output = _dense(8, 2)
#     total = bilstm + dense1 + output

#     return {
#         'input_dim': input_dim,
#         'sequence_length': seq_len,
#         'bilstm_fwd_8': forward,
#         'bilstm_bwd_8': backward,
#         'bilstm_total': bilstm,
#         'dense_8': dense1,
#         'output': output,
#         'total': total,
#     }


# def build_summary(input_dims=INPUT_DIMS):
#     """
#     Build summary DataFrames for parameter comparison.
    
#     Returns:
#         df_wide: One row per input_dim, columns per model
#         df_long: Tidy format for seaborn plots
#     """
#     wide_rows = []
#     long_rows = []

#     for d in input_dims:
#         rq = params_InLoq(d)
#         target = rq['total']
#         an = params_ann(d, target_params=target)
#         ls = params_lstm(d)
#         gr = params_gru(d)
#         bl = params_bilstm(d)

#         wide_rows.append({
#             'Input_Dim': d,
#             'InLoq_Quantum_Only': rq['quantum'],
#             'InLoq_Classical_Only': rq['classical_only'],
#             'InLoq_Total': rq['total'],
#             'ANN_Total': an['total'],
#             'LSTM_Total': ls['total'],
#             'GRU_Total': gr['total'],
#             'BiLSTM_Total': bl['total'],
#         })

#         for model_name, total in [
#             ('InLoq (Proposed)', rq['total']),
#             ('Classical ANN', an['total']),
#             ('LSTM', ls['total']),
#             ('GRU', gr['total']),
#             ('BiLSTM', bl['total']),
#         ]:
#             long_rows.append({
#                 'Input Dimension': d,
#                 'Model': model_name,
#                 'Parameters': total,
#             })

#         # Extra row for quantum portion only
#         long_rows.append({
#             'Input Dimension': d,
#             'Model': 'InLoq (quantum only)',
#             'Parameters': rq['quantum'],
#         })

#     df_wide = pd.DataFrame(wide_rows)
#     df_long = pd.DataFrame(long_rows)
#     return df_wide, df_long