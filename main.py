from qml_inference_circuit import *

x = array([0.75, 0.5, 1])
w = array([15.0406, -20.47, 1.70474])
inner_prod_bit_accuracy = 3
activation_fxn_bit_accuracy = 1
backend_name = "ibmq_qasm_simulator"


def relu(z):
    return max(0, z)


def sigmoid(z):
    return 1 / (1 + exp(-z))


def activation_function(z):
    return sigmoid(z)


print(dot(x, w))
print(activation_function(dot(x, w)))

circuit = inference_circuit()
circuit.encode_data(x)
circuit.add_inner_product_module(w, bit_accuracy=inner_prod_bit_accuracy)
circuit.add_activation_fxn_module(activation_function, bit_accuracy=activation_fxn_bit_accuracy)
circuit.draw_circuit()
circuit.execute_circuit(shots=20000, backend=backend_name, optimization_level=None)
circuit.display_results()
