from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile
from qiskit.circuit.library import Diagonal, CPhaseGate

from qiskit.providers.ibmq import least_busy
from qiskit import IBMQ

from numpy import array, exp, pi
import matplotlib.pyplot as plt


class inference_circuit:
    """"""

    def __init__(self):
        self.basis_encoding_hf = self.basis_encoding()
        self.algorithms_hf = self.algorithms()
        self.inference_circuit = QuantumCircuit()
        self.num_of_qubits = 0

        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='ibm-q')

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

        num_of_ip_anc = len(self.inner_prod_reg)
        qregs = self.inference_circuit.qregs
        x_qregs = array([reg for reg in qregs if "x" in reg.name])
        print(x_qregs)

        x_qregs_sorted = []
        temp = []
        curr = 0
        for i in range(len(x_qregs)):
            for x in x_qregs:
                if "{}".format(curr) in x[0].register.name:
                    temp += [x]
            x_qregs_sorted += [temp]
            curr += 1
        print(x_qregs_sorted)

        w = 0
        for x in x_qregs_sorted:
            for type in x:
                for qubit in type:
                    for i in range(num_of_ip_anc):
                        if "int" in qubit.register.name:
                            self.inference_circuit.append(
                                CPhaseGate(w_vector[w] * (2 ** qubit.index) * pi / 2 ** i),
                                [qubit.register[qubit.index], self.inner_prod_reg[i]])
                        if "float" in qubit.register.name:
                            self.inference_circuit.append(
                                CPhaseGate(w_vector[w] * (2 ** -(qubit.index + 1)) * pi / 2 ** i),
                                [qubit.register[qubit.index], self.inner_prod_reg[i]])
            w += 1

        self.algorithms_hf.inv_qft(self.inference_circuit, self.inner_prod_reg)

    def add_activation_fxn_module(self, fxn, bit_accuracy):
        """"""
        self.activation_fxn_reg = QuantumRegister(bit_accuracy, 'anc_fxn')
        self.inference_circuit.add_register(self.activation_fxn_reg)
        self.algorithms_hf.create_superposition(self.inference_circuit, self.activation_fxn_reg)

        def diag_element(z):
            return exp(2 * pi * 1j * fxn(z) / 2 ** bit_accuracy)

        D_gate = Diagonal(
            array([diag_element(0), diag_element(1), diag_element(-2), diag_element(-1)])
        )

        for bit in range(1, bit_accuracy + 1):
            self.inference_circuit.append(D_gate.control(1).power(bit),
                                          [self.activation_fxn_reg[bit - 1]] + self.inner_prod_reg[:])

        self.algorithms_hf.inv_qft(self.inference_circuit, self.activation_fxn_reg)

    def draw_circuit(self):
        """"""
        self.inference_circuit.draw(output='mpl')
        plt.show()

    def measure_register(self, register):
        classical_reg = ClassicalRegister(len(register), 'result')
        self.inference_circuit.add_register(classical_reg)
        self.inference_circuit.measure(register, classical_reg)

    def execute_circuit(self, shots, backend=None, optimization_level=None):
        """"""
        self.measure_register(self.activation_fxn_reg)
        self.get_number_of_qubits()

        if backend == None:
            pass
        # self.backend = least_busy(
        #     self.provider.backends(filters=lambda x: x.configuration().n_qubits >= self.num_of_qubits
        #                                              and not x.configuration().simulator
        #                                              and x.status().operational == True)
        # )
        else:
            self.backend = self.provider.get_backend(backend)
            job = execute(
                transpile(self.inference_circuit, backend=backend, optimization_level=optimization_level),
                backend=self.backend,
                shots=shots
            )
            self.result = job.result()

    def display_results(self):
        """"""
        print(self.result)

    def get_backend_data(self, backend):
        """"""
        print(self.result)

    def get_number_of_qubits(self):
        self.num_of_qubits = self.inference_circuit.num_qubits
        print("Number of Qubits Required: ", self.num_of_qubits)

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

        def qft(self, circuit, q_reg):
            self.qft_rotations(circuit, len(q_reg))

        def qft_rotations(self, circuit, qubit_number):
            if qubit_number == 0:
                return circuit
            qubit_number -= 1
            circuit.h(qubit_number)
            for qubit in range(qubit_number):
                circuit.cp(pi / 2 ** (qubit_number - qubit), qubit, qubit_number)

            self.qft_rotations(circuit, qubit_number)

        def inv_qft(self, circuit, q_reg):
            dummy_circuit = QuantumCircuit(len(q_reg))
            self.qft(dummy_circuit, q_reg)
            invqft_circuit = dummy_circuit.inverse()
            circuit.append(invqft_circuit, q_reg)
