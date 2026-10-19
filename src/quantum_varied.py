# src/quantum_varied.py
"""
Variable-depth quantum layer for ablation studies.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import pennylane as qml
import numpy as np


def _build_qnode_variable_depth(n_qubits: int, n_layers: int):
    """QNode with a configurable number of variational layers."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def circuit(inputs, weights):
        inputs = tf.cast(inputs, tf.float32)
        if inputs.shape[-1] is not None:
            if inputs.shape[-1] < n_qubits:
                pad = tf.zeros((*inputs.shape[:-1], n_qubits - inputs.shape[-1]),
                              dtype=tf.float32)
                inputs = tf.concat([inputs, pad], axis=-1)
            elif inputs.shape[-1] > n_qubits:
                inputs = inputs[..., :n_qubits]
        
        qml.templates.AngleEmbedding(inputs * np.pi,
                                     wires=range(n_qubits), rotation='Y')
        
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1],
                        weights[l, i, 2], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class QuantumLayerVarDepth(Layer):
    """
    Quantum layer with configurable number of variational layers.
    Used ONLY for ablation studies (not the main model).
    """

    def __init__(self, n_qubits: int, input_dim: int,
                 n_layers: int = 2, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.n_layers = n_layers
        self._seed = seed
        self.circuit = _build_qnode_variable_depth(n_qubits, n_layers)

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name='q_weights',
            shape=(self.n_layers, self.n_qubits, 3),
            initializer=tf.keras.initializers.GlorotUniform(seed=self._seed),
            trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        out = self.circuit(inputs, self.q_weights)
        return tf.cast(tf.math.real(tf.stack(out, axis=-1)), tf.float32)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'n_qubits': self.n_qubits,
            'input_dim': self.input_dim,
            'n_layers': self.n_layers,
            'seed': self._seed
        })
        return cfg