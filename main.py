from qml_inference_circuit import *

# x = array([1, 0.75, 1])
# w = array([15.0406, -20.47, 1.70474])
# inner_prod_bit_accuracy = 3
# activation_fxn_bit_accuracy = 1
backend_name = "ibmq_qasm_simulator"
import numpy as np


def relu(z):
    return max(0, z)


def sigmoid(z):
    return 1 / (1 + exp(-z))


def activation_function(z):
    return relu(z)


def linear(z):
    return z


#
#
# print("Classical Dot Product: ", dot(x, w))
# print("Sigmoid f(w*x) = ", activation_function(dot(x, w)))
#
# circuit = basis_encoding_circuit()
# circuit.encode_data(x)
# circuit.add_inner_product_module(w, bit_accuracy=inner_prod_bit_accuracy)
# circuit.add_activation_fxn_module(activation_function, bit_accuracy=activation_fxn_bit_accuracy)
# circuit.draw_circuit()
# circuit.execute_circuit(shots=20000, backend=backend_name, optimization_level=None)
# circuit.display_results()
# circuit.get_circuit_data()
# circuit.get_backend_data()

x = array([1, 1])
w = array([1, 1])

circuit = amplitude_encoding_circuit()
circuit.encode_data(x, w)
circuit.build_circuit(qft_bit_accuracy=2)
# circuit.add_activation_fxn_module(activation_function, bit_accuracy=2)
circuit.draw_circuit()

circuit.execute_circuit(shots=20000, backend=backend_name, optimization_level=None)
