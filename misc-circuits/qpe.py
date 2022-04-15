# This example is from Qiskit's QPE tutorial: https://qiskit.org/textbook/ch-algorithms/quantum-phase-estimation.html

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.visualization import plot_histogram
from qiskit import IBMQ
import os
import matplotlib.pyplot as plt
from numpy import pi

circuit = QuantumCircuit(4, 3)

for qubit in range(3):
    circuit.h(qubit)

circuit.x(3)

# FROM QISKIT's TUTORIAL ON QPE
repetitions = 1
for counting_qubit in range(3):
    for i in range(repetitions):
        circuit.cp(pi / 4, counting_qubit, 3);  # This is CU
    repetitions *= 2


def qft_dagger(qc, n):
    # FROM QISKIT's TUTORIAL ON QPE
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-pi / float(2 ** (j - m)), m, j)
        qc.h(j)


# Apply inverse QFT
qft_dagger(circuit, 3)

for n in range(3):
    circuit.measure(n, n)

os.chdir('../circuit-photos')
path = os.getcwd()
circuit.draw('mpl', fold=30, filename=path + '/qpe_example_circuit.png', style={'name': 'bw', 'dpi': 350})
plt.show()

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
job = execute(circuit, backend=provider.get_backend("ibmq_qasm_simulator"), shots=10000)
os.chdir('../results-photos')
path = os.getcwd()
plot_histogram(job.result().get_counts(), filename=path + '/qpe_example_results.png', color='black',
               title="QPE Measurement Outcome")
plt.show()
