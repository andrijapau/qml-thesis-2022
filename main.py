from qml_inference_circuit import *

x = array([1])
w = array([1 / 0.75])
inner_prod_bit_accuracy = 2
activation_fxn_bit_accuracy = 1
backend_name = "ibmq_qasm_simulator"


def relu(z):
    return max(0, z)


circuit = inference_circuit()
circuit.encode_data(x)
circuit.add_inner_product_module(w, bit_accuracy=inner_prod_bit_accuracy)
circuit.add_activation_fxn_module(relu, bit_accuracy=activation_fxn_bit_accuracy)
circuit.execute_circuit(shots=20000, backend=backend_name, optimization_level=None)
circuit.display_results()
circuit.draw_circuit()
