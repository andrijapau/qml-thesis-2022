from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile
from qiskit.circuit.library import Diagonal

from numpy import array, exp, pi
import matplotlib.pyplot as plt


class inference_circuit:
    """"""

    def __init__(self):
        self.basis_encoding_hf = self.basis_encoding()
        self.algorithms_hf = self.algorithms()
        self.inference_circuit = QuantumCircuit()

    def encode_data(self, x_vector):
        """"""
        data_num = 0
        for x in x_vector:
            self.basis_encoding_hf.embed_data_to_circuit(x, self.inference_circuit, data_num=data_num)
            data_num += 1

    def add_inner_product_module(self, w_vector, bit_accuracy):
        """"""
        self.inner_prod_reg = QuantumRegister(bit_accuracy, 'anc_ip')
        self.inference_circuit.add_register(self.inner_prod_reg)
        self.algorithms_hf.create_superposition(self.inference_circuit, self.inner_prod_reg)

    def add_activation_fxn_module(self, fxn, bit_accuracy):
        """"""
        self.activation_fxn_reg = QuantumRegister(bit_accuracy, 'anc_fxn')
        self.inference_circuit.add_register(self.activation_fxn_reg)
        self.algorithms_hf.create_superposition(self.inference_circuit, self.activation_fxn_reg)

        def diag_element(z):
            return exp(2 * pi * 1j * z / 2 ** bit_accuracy)

        D_gate = Diagonal(
            array([diag_element(0), diag_element(1), diag_element(-2), diag_element(-1)])
        ).control(1).decompose()

        self.inference_circuit.append(D_gate, self.inner_prod_reg[:] + self.activation_fxn_reg[:])

    def draw_circuit(self):
        """"""
        self.inference_circuit.draw(output='mpl')
        plt.show()

    def execute_circuit(self, backend, shots, optimization_level=None):
        """"""
        classical_reg = ClassicalRegister(len(self.activation_fxn_reg), 'result')
        self.inference_circuit.add_register(classical_reg)
        self.inference_circuit.measure(self.activation_fxn_reg, classical_reg)
        print(self.inference_circuit.qregs)
        # job = execute(
        #     transpile(self.circuit, backend=backend, optimization_level=optimization_level),
        #     backend=backend,
        #     shots=shots
        # )
        # self.result = job.result()

    def display_results(self):
        """"""
        print(self.result)

    def get_backend_data(self, backend):
        """"""
        print(self.result)

    class basis_encoding:
        """"""

        def __init__(self):
            self.int_bin, self.float_bin = '', ''

        def convert_data_to_binary(self, decimal):
            ''''''
            self.int_bin = '1'
            self.float_bin = ''

        def embed_data_to_circuit(self, data, circuit, data_num=None):
            """"""
            self.convert_data_to_binary(data)

            if len(self.int_bin) != 0:
                int_reg = QuantumRegister(len(self.int_bin), 'x{}_int'.format(data_num))
                circuit.add_register(int_reg)
                for i in range(len(self.int_bin)):
                    if self.int_bin[i] == '1':
                        circuit.x(int_reg[i])

            if len(self.float_bin) != 0:
                float_reg = QuantumRegister(len(self.float_bin), 'x{}_float'.format(data_num))
                circuit.add_register(float_reg)
                for i in range(len(self.float_bin)):
                    if self.float_bin[i] == '1':
                        circuit.x(float_reg[i])

    class algorithms:
        """"""

        def __init__(self):
            print(1)

        def create_superposition(self, circuit, register):
            """"""
            circuit.h(register)

        def inverse_qft(self, circuit, register):
            """"""
            print(1)
