from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.visualization import plot_histogram
from qiskit import IBMQ
import os
import matplotlib.pyplot as plt
from numpy import pi


def qft(circuit, q_reg):
    qft_rotations(circuit, len(q_reg))


def qft_rotations(circuit, qubit_number):
    if qubit_number == 0:
        return circuit
    qubit_number -= 1
    circuit.h(qubit_number)
    for qubit in range(qubit_number):
        circuit.cp(pi / 2 ** (qubit_number - qubit), qubit, qubit_number)

    qft_rotations(circuit, qubit_number)


def inv_qft(circuit, q_reg):
    dummy_circuit = QuantumCircuit(len(q_reg), name=r'$QFT^\dagger$')
    qft(dummy_circuit, q_reg)
    invqft_circuit = dummy_circuit.inverse().decompose()
    circuit.append(invqft_circuit.decompose(), q_reg)
    return circuit


num_of_qubits = 3
num_of_classical_bits = 3

q_reg = QuantumRegister(num_of_qubits)
c_reg = ClassicalRegister(num_of_classical_bits)

circuit = QuantumCircuit(q_reg, c_reg)

circuit.x(q_reg[0])
circuit.i(q_reg[1])
circuit.x(q_reg[2])

circuit.barrier()

qft(circuit, q_reg)
circuit.barrier()
# circuit = inv_qft(circuit, q_reg)
# circuit.barrier()

circuit.measure(q_reg, c_reg)

os.chdir('../circuit-photos')
path = os.getcwd()
circuit.draw('mpl', filename=path + '/qft_example_circuit.png', style={'name': 'bw', 'dpi': 350})
plt.show()

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
job = execute(circuit, backend=provider.get_backend("ibmq_qasm_simulator"), shots=10000)

os.chdir('../results-photos')
path = os.getcwd()
plot_histogram(job.result().get_counts(), filename=path + '/qft_example_circuit_results.png', color='black',
               title="Circuit Measurement Outcome")
plt.show()
