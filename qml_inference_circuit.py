from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile
from numpy import array
import matplotlib.pyplot as plt


class inference_circuit:
    def __init__(self):
        self.basis_encoding_hf = self.basis_encoding()
        self.algorithms_hf = self.algorithms()

        self.inference_circuit = QuantumCircuit()

    def encode_data(self, x_vector):
        data_num = 0
        for x in x_vector:
            self.basis_encoding_hf.embed_data_to_circuit(x, self.inference_circuit, data_num=data_num)
            data_num += 1

    def add_inner_product_module(self, w_vector, bit_accuracy):
        self.inner_prod_reg = QuantumRegister(bit_accuracy, 'anc_ip')
        self.inference_circuit.add_register(self.inner_prod_reg)
        self.algorithms_hf.create_superposition(self.inference_circuit, self.inner_prod_reg)

    def add_activation_fxn_module(self, fxn, bit_accuracy):
        self.activation_fxn_reg = QuantumRegister(bit_accuracy, 'anc_fxn')
        self.inference_circuit.add_register(self.activation_fxn_reg)
        self.algorithms_hf.create_superposition(self.inference_circuit, self.activation_fxn_reg)

    def draw_circuit(self):
        self.inference_circuit.decompose().draw(output='mpl')
        plt.show()

    def execute_circuit(self, backend, shots, optimization_level=None):
        classical_reg = ClassicalRegister(len(self.activation_fxn_reg))
        self.inference_circuit.measure(self.activation_fxn_reg, classical_reg)
        job = execute(
            transpile(self.circuit, backend=backend, optimization_level=optimization_level),
            backend=backend,
            shots=shots
        )
        self.result = job.result()

    def display_results(self):
        print(self.result)

    class basis_encoding:
        def __init__(self):
            print(1)

        def convert_data_to_binary(self, decimal):
            int_bin = '1'
            float_bin = ''

            return int_bin, float_bin

        def create_data_circuit(self, int_bin, float_bin):
            data_circuit = QuantumCircuit()

            if len(int_bin) != 0:
                int_reg = QuantumRegister(len(int_bin), 'x_int')
                data_circuit.add_register(int_reg)
                for i in range(len(int_bin)):
                    if int_bin[i] == '1':
                        data_circuit.x(int_reg[i])

            if len(float_bin) != 0:
                float_reg = QuantumRegister(len(float_bin), 'x_float')
                data_circuit.add_register(float_reg)
                for i in range(len(float_bin)):
                    if float_bin[i] == '1':
                        data_circuit.x(float_reg[i])

            return data_circuit

        def embed_data_to_circuit(self, data, circuit, data_num=None):
            int_bin, float_bin = self.convert_data_to_binary(data)
            data_circuit = self.create_data_circuit(int_bin, float_bin)

            x_reg = QuantumRegister(len(int_bin) + len(float_bin), 'x{}'.format(data_num))
            circuit.add_register(x_reg)
            circuit.append(data_circuit, x_reg)

    class algorithms:
        def __init__(self):
            print(1)

        def create_superposition(self, circuit, register):
            circuit.h(register)

        def inverse_qft(self, circuit, register):
            print(1)
