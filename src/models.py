"""
Model definitions: QuantumLayer, InLoq, ANN, LSTM, GRU, BiLSTM
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, Input, Layer
from tensorflow.keras.optimizers import Adam
import pennylane as qml
import numpy as np

from src.utils import set_global_seeds, count_params, SEQUENCE_LENGTH
from src.utils import set_global_seeds, count_params


def _ki(seed):
    return tf.keras.initializers.GlorotUniform(seed=seed)


def _oi(seed):
    return tf.keras.initializers.Orthogonal(seed=seed)


def _build_qnode(n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def circuit(inputs, weights):
        inputs = tf.cast(inputs, tf.float32)
        if inputs.shape[-1] is not None:
            if inputs.shape[-1] != n_qubits:
                pad = tf.zeros((*inputs.shape[:-1], n_qubits - inputs.shape[-1]),
                               dtype=tf.float32)
                inputs = tf.concat([inputs, pad], axis=-1)
            elif inputs.shape[-1] > n_qubits:
                inputs = inputs[..., :n_qubits]
        qml.templates.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
        for l in range(2):
            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class QuantumLayer(Layer):
    """Keras layer wrapping a PennyLane QNode with deterministic init."""

    def __init__(self, n_qubits: int, input_dim: int, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self._seed = seed
        self.circuit = _build_qnode(n_qubits)

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name='q_weights', shape=(2, self.n_qubits, 3),
            initializer=tf.keras.initializers.GlorotUniform(seed=self._seed),
            trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        out = self.circuit(inputs, self.q_weights)
        return tf.cast(tf.math.real(tf.stack(out, axis=-1)), tf.float32)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'n_qubits': self.n_qubits,
                    'input_dim': self.input_dim, 'seed': self._seed})
        return cfg


def build_InLoq(input_dim: int, seed: int = 42) -> Sequential:
    """Build InLoQ (Quantum Neural Network) model."""
    set_global_seeds(seed)
    n_q = min(max(input_dim, 2), 8)
    m = Sequential([
        Dense(n_q, input_dim=input_dim, activation='tanh', name='classical_pre',
              kernel_initializer=_ki(seed)),
        QuantumLayer(n_q, n_q, seed=seed, name='quantum_layer'),
        Dense(16, activation='relu', name='post_q1', kernel_initializer=_ki(seed+1)),
        Dense(8, activation='relu', name='post_q2', kernel_initializer=_ki(seed+2)),
        Dense(2, activation='linear', name='output', kernel_initializer=_ki(seed+3)),
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m


def build_ann(input_dim: int, seed: int = 42, width: int = 32) -> Sequential:
    """Build Classical ANN model."""
    set_global_seeds(seed)
    m = Sequential([
        Dense(width, input_dim=input_dim, activation='relu',
              name='layer1', kernel_initializer=_ki(seed)),
        Dense(width // 2, activation='relu', name='layer2', kernel_initializer=_ki(seed+1)),
        Dense(2, activation='linear', name='output', kernel_initializer=_ki(seed+2)),
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m


def build_lstm(input_dim: int, seq_len: int = 6, seed: int = 42) -> Sequential:
    """Build LSTM model."""
    set_global_seeds(seed)
    m = Sequential([
        Input(shape=(seq_len, input_dim)),
        LSTM(16, return_sequences=False,
             kernel_initializer=_ki(seed), recurrent_initializer=_oi(seed)),
        Dropout(0.1, seed=seed),
        Dense(8, activation='relu', kernel_initializer=_ki(seed+1)),
        Dense(2, activation='linear', kernel_initializer=_ki(seed+2)),
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m


def build_gru(input_dim: int, seq_len: int = 6, seed: int = 42) -> Sequential:
    """Build GRU model."""
    set_global_seeds(seed)
    m = Sequential([
        Input(shape=(seq_len, input_dim)),
        GRU(16, return_sequences=False,
            kernel_initializer=_ki(seed), recurrent_initializer=_oi(seed)),
        Dropout(0.1, seed=seed),
        Dense(8, activation='relu', kernel_initializer=_ki(seed+1)),
        Dense(2, activation='linear', kernel_initializer=_ki(seed+2)),
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m


def build_bilstm(input_dim: int, seq_len: int = 6, seed: int = 42) -> Sequential:
    """Build Bidirectional LSTM model."""
    set_global_seeds(seed)
    m = Sequential([
        Input(shape=(seq_len, input_dim)),
        Bidirectional(LSTM(8, return_sequences=False,
                           kernel_initializer=_ki(seed),
                           recurrent_initializer=_oi(seed))),
        Dropout(0.1, seed=seed),
        Dense(8, activation='relu', kernel_initializer=_ki(seed+1)),
        Dense(2, activation='linear', kernel_initializer=_ki(seed+2)),
    ])
    m.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return m


# def get_model_builder(model_name: str):
#     """Return model builder function for given model name."""
#     builders = {
#         'Inloq': build_InLoq,
#         'ANN': build_ann,
#         'LSTM': build_lstm,
#         'GRU': build_gru,
#         'BiLSTM': build_bilstm,
#     }
#     return builders[model_name]

def build_model(name: str, input_dim: int, seed: int):
    if name == 'Inloq':   return build_InLoq(input_dim, seed)
    if name == 'ANN':     return build_ann(input_dim, seed)
    if name == 'LSTM':    return build_lstm(input_dim, SEQUENCE_LENGTH, seed)
    if name == 'GRU':     return build_gru(input_dim, SEQUENCE_LENGTH, seed)
    if name == 'BiLSTM':  return build_bilstm(input_dim, SEQUENCE_LENGTH, seed)
    raise ValueError(f'Unknown model: {name}')
