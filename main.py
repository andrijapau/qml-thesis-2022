# Import model
from model_testing import *

# Define bit accuracy for various computations
inner_prod_bit_accuracy = 2
activation_fxn_bit_accuracy = 1

# Define IBMQ Quantum backend to use
backend_name = "ibm_nairobi"

# Trained weight parameters: (w1, w2, b)
scale = 1 / 10
beta = array([-8.803, -14.53, 12.57]) * scale

# Create model and execute on backend to display results
test = model_testing(trained_weights=beta, inner_product_bit_accuracy=inner_prod_bit_accuracy,
                     activation_fxn_bit_accuracy=activation_fxn_bit_accuracy, debug=True)
test.load_csv("./datasets/qml_dataset.csv")
test.run_model_on_backend(backend=backend_name, multipleShots=True, withMitigation=False, seed=1005)
test.print_metrics()

# AMPLITUDE ENCODING STUFF BELOW.

# test.print_backend_info()
# def sigmoid(z):
#     return 1 / (1 + exp(-z))
#
#
# def activation_function(z):
#     return sigmoid(z)
# print("Classical Dot Product: ", dot(x, w))
# print("Sigmoid f(w*x) = ", activation_function(dot(x, w)))
# circuit = basis_encoding_circuit()
# circuit.encode_data(x)
# circuit.add_inner_product_module(w, bit_accuracy=inner_prod_bit_accuracy)
# circuit.add_activation_fxn_module(activation_function, bit_accuracy=activation_fxn_bit_accuracy)
# circuit.draw_circuit()
# circuit.execute_circuit(shots=20000, backend=backend_name, optimization_level=None)
# circuit.display_results()
# circuit.get_circuit_data()
# circuit.get_backend_data()
#
# x = array([1, 1])
# w = array([1, 1])
#
# circuit = amplitude_encoding_circuit()
# circuit.encode_data(x, w)
# circuit.build_circuit(qft_bit_accuracy=2)
# # circuit.add_activation_fxn_module(activation_function, bit_accuracy=2)
# circuit.draw_circuit()
#
# circuit.execute_circuit(shots=20000, backend=backend_name, optimization_level=None)
