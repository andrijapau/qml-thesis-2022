from qml_inference_circuit import *
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import OrderedDict
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, CompleteMeasFitter)
from qiskit import Aer
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error

from numpy import where, array
import csv
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame


class model_testing:
    def __init__(self, trained_weights=None, inner_product_bit_accuracy=None, activation_fxn_bit_accuracy=None,
                 debug=False):
        self.debug = debug
        self.beta = trained_weights
        self.ip_bit_accuracy = inner_product_bit_accuracy
        self.fxn_bit_accuracy = activation_fxn_bit_accuracy
        self.scaling_factor = 10
        self.f = lambda z: 1 / (1 + exp(-1 * self.scaling_factor * z))

        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='strangeworks-hub', group="science-team", project="science-test")

        self.avg_num_of_qubits = []
        self.avg_gate_depth = []
        self.avg_number_of_gates = []

    def load_csv(self, csv_name=None):
        with open(csv_name, "r") as csvfile:
            lines = csv.reader(csvfile)
            dataset = list(lines)
            for i in range(len(dataset)):
                dataset[i] = [float(x) for x in dataset[i]]

        # def normalize_feature_matrix(X):
        #     '''
        #     function to normalize feature matrix, X
        #     '''
        #     mins = min(X, axis=0)
        #     maxs = max(X, axis=0)
        #     rng = maxs - mins
        #     norm_X = 1 - ((maxs - X) / rng)
        #     return norm_X

        self.dataset = array(dataset)
        # self.x_test = normalize_feature_matrix(dataset[:, :-1])
        self.x_test = self.dataset[:, :-1]
        self.y_true = self.dataset[:, -1]

        self.y_pred_shots = []
        self.y_pred = []
        self.y_simulator_pred = []

    def run_model_on_backend(self, backend, shots=100000, multipleShots=False, withMitigation=False, seed=None):

        # Get backend from name
        self.backend_name = backend
        self.backend = self.provider.get_backend(backend)

        # Define circuit batch list for dataset
        circuit_batch_list = []

        # Loop through all test vectors and build a circuit for each one
        for i in range(len(self.x_test)):

            if i % 2 == 0:
                print("Building circuit(s) ... {} %".format(round(100 * i / len(self.x_test), 3)))

            # Build circuit
            self.circuit = basis_encoding_circuit()
            self.circuit.encode_data(self.x_test[i])
            self.circuit.add_inner_product_module(self.beta, bit_accuracy=self.ip_bit_accuracy)
            self.circuit.add_activation_fxn_module(self.f, bit_accuracy=self.fxn_bit_accuracy)

            # Build filtered circuit that removes all idle wires
            self.filtered_circuit_temp = self.remove_idle_qwires(self.circuit.inference_circuit)
            self.filtered_circuit_temp.qasm(filename='filtered')
            self.filtered_circuit = QuantumCircuit.from_qasm_file('filtered')

            if self.debug:
                print("Dot product result: ", dot(self.x_test[i], self.beta))
                print("Sigmoid output: ", self.f(dot(self.x_test[i], self.beta)))
                self.circuit.draw_circuit()
                self.filtered_circuit.draw(output='mpl')
                plt.show()

            # Only execute circuits that have less qubits than the maximum amount on the backend
            backend_num_of_qubits = 7
            if self.filtered_circuit.num_qubits <= backend_num_of_qubits:

                # If filtered circuit contains less qubits than the original circuit than it removed idle wires
                if self.filtered_circuit.num_qubits < self.circuit.inference_circuit.num_qubits:

                    # Append filtered circuit to batch
                    circuit_batch_list.append(self.filtered_circuit)

                    # Track circuit metrics
                    self.avg_num_of_qubits.append(self.filtered_circuit.num_qubits)
                    self.avg_gate_depth.append(self.filtered_circuit.depth())

                # Idle wires were not removed
                else:

                    # Append circuit to batch
                    circuit_batch_list.append(self.circuit.inference_circuit)

                    # Track circuit metrics
                    self.avg_num_of_qubits.append(self.circuit.inference_circuit.num_qubits)
                    self.avg_gate_depth.append(self.circuit.inference_circuit.depth())

            # Print to user which test vectors required too many qubits.
            else:
                print("This vector needs too many qubits: {}".format(self.x_test[i]))
        print("Building circuit(s) ... Done \n")

        # If user would like to test multiple shots
        if multipleShots:

            self.shots_values = [1e1, 1e2, 1e3, 1e4, 1e5]
            # self.shots_values = [1e3] * 30
            for shot in self.shots_values:

                print("Running circuit(s) on {} for {} shots (on seed = {})... ".format(self.backend_name, shot, seed))
                job = execute(
                    transpile(circuit_batch_list, self.backend, seed_transpiler=seed, optimization_level=3),
                    backend=self.backend,
                    shots=int(shot)
                )
                print("Done. \n")

                self.result = job.result()
                temp = []

                for i in range(len(circuit_batch_list)):
                    if withMitigation:
                        print("Running mitigation circuits for {} shots... ".format(shot))
                        cal_circuits, state_labels = complete_meas_cal(qr=circuit_batch_list[i].data.pop(),
                                                                       circlabel='measerrormitigationcal')
                        cal_job = execute(cal_circuits, backend=self.backend, shots=50000, optimization_level=3)
                        cal_results = cal_job.result()
                        meas_fitter = CompleteMeasFitter(cal_results, state_labels)
                        self.meas_filter = meas_fitter.filter
                        specific_result = job.result(i)
                        self.mitigated_result = self.meas_filter.apply(specific_result)
                        counts = self.mitigated_result.get_counts(i)

                    else:
                        counts = self.result.get_counts(i)

                    y_pred_value = self.get_class(counts)
                    temp.append([y_pred_value])
                self.y_pred_shots.append(temp)

        else:

            if withMitigation:
                cal_circuits, state_labels = complete_meas_cal(qr=self.circuit.inference_circuit.qregs[0],
                                                               circlabel='measerrormitigationcal')
                cal_job = execute(cal_circuits, backend=self.backend, shots=shots, optimization_level=3,
                                  seed_transpiler=seed)
                cal_results = cal_job.result()
                meas_fitter = CompleteMeasFitter(cal_results, state_labels)
                self.meas_filter = meas_fitter.filter
            print("Running circuit(s) on {} ... ".format(self.backend_name))
            job = execute(
                transpile(circuit_batch_list, self.backend, seed_transpiler=seed, optimization_level=3),
                backend=self.backend,
                shots=shots
            )
            self.result = job.result()
            print("Done. \n")

            for i in range(len(circuit_batch_list)):
                if withMitigation:
                    counts = self.mitigated_result.get_counts(i)
                else:
                    counts = self.result.get_counts(i)
                y_pred_value = self.get_class(counts)
                self.y_pred.append([y_pred_value])

        # Run results on simulator to track how it compares
        if self.backend_name != "ibmq_qasm_simulator":

            print("Running circuit(s) on ibmq_qasm_simulator ... ")
            self.sim_backend = self.provider.get_backend("ibmq_qasm_simulator")
            job = execute(
                transpile(circuit_batch_list, self.sim_backend),
                backend=self.sim_backend,
                shots=shots
            )
            print("Done. \n")
            self.simulation_results = job.result()

            for i in range(len(circuit_batch_list)):
                counts = self.simulation_results.get_counts(i)
                y_simulation_pred_value = self.get_class(counts)
                self.y_simulator_pred.append([y_simulation_pred_value])

    def print_metrics(self):
        print("Printing metrics for backend {} ... \n".format(self.backend_name))

        self.print_confusion_matrix_and_measures()
        # self.print_auc_score()
        self.plot_roc_curve()

        print("Max number of qubits: {}".format(max(self.avg_num_of_qubits)))
        print("Max gate depth: {}".format(max(self.avg_gate_depth)))

    def print_backend_info(self):
        plot_gate_map(self.backend)
        plot_error_map(self.backend)
        # plot_coupling_map(self.backend)
        plt.show()

    def get_class(self, counts):
        total_shots = sum(list(counts.values()))
        y_pred = 0
        for key, value in counts.items():
            if key == "1":
                y_pred = where(value / total_shots >= .5, 1, 0)
        return y_pred

    def print_confusion_matrix_and_measures(self):
        if self.y_pred_shots == 0:
            self.cfn_matrix = confusion_matrix(self.y_true, self.y_pred)
            print(self.cfn_matrix)
            TP = self.cfn_matrix[0, 0]  # True Positive
            FP = self.cfn_matrix[0, 1]  # False Positive
            FN = self.cfn_matrix[1, 0]  # False Negative
            TN = self.cfn_matrix[1, 1]  # True Negative

            def precision(TP, FP):
                return TP / (TP + FP)

            def recall(TP, FN):
                return TP / (TP + FN)

            def accuracy(TP, TN, FP, FN):
                return (TP + TN) / (TP + TN + FP + FN)

            def F_measure(TP, FP, FN):
                return (2 * TP) / (2 * TP + FP + FN)

            print("CONFUSION MATRIX: \n", self.cfn_matrix)
            print("Precision = ", precision(TP, FP))
            print("Recall = ", recall(TP, FN))
            print("Accuracy = ", accuracy(TP, TN, FP, FN))
            print("F1 = ", F_measure(TP, FP, FN))
        else:
            for i in range(len(self.y_pred_shots)):
                self.cfn_matrix = confusion_matrix(self.y_true, self.y_pred_shots[i])

                TP = self.cfn_matrix[0, 0]  # True Positive
                FP = self.cfn_matrix[0, 1]  # False Positive
                FN = self.cfn_matrix[1, 0]  # False Negative
                TN = self.cfn_matrix[1, 1]  # True Negative

                def precision(TP, FP):
                    return TP / (TP + FP)

                def recall(TP, FN):
                    return TP / (TP + FN)

                def accuracy(TP, TN, FP, FN):
                    return (TP + TN) / (TP + TN + FP + FN)

                def F_measure(TP, FP, FN):
                    return (2 * TP) / (2 * TP + FP + FN)

                print("CONFUSION MATRIX for {} shots: {}\n".format(self.shots_values[i], self.cfn_matrix))
                print("Precision for {} shots = {}".format(self.shots_values[i], precision(TP, FP)))
                print("Recall for {} shots = {}".format(self.shots_values[i], recall(TP, FN)))
                print("Accuracy for {} shots = {}".format(self.shots_values[i], accuracy(TP, TN, FP, FN)))
                print("F1 for {} shots = {}".format(self.shots_values[i], F_measure(TP, FP, FN)))

    def print_auc_score(self):
        if len(self.y_pred_shots) != 0:
            for i in range(len(self.y_pred_shots)):
                auc_score = roc_auc_score(self.y_true, self.y_pred_shots[i])
                print("AUC for {} shots = {}".format(self.shots_values[i], auc_score))
        else:
            auc_score = roc_auc_score(self.y_true, self.y_pred)
            print("AUC = {}".format(auc_score))

    def plot_roc_curve(self):
        if len(self.y_pred_shots) != 0:
            for i in range(len(self.y_pred_shots)):
                fpr, tpr, thres = roc_curve(self.y_true, self.y_pred_shots[i], pos_label=1)
                plt.plot(fpr, tpr, linestyle='-', label='{:.1e} (shots)'.format(self.shots_values[i]))
                # if i == 0:
                #     fpr, tpr, thres = roc_curve(self.y_true, self.y_pred_shots[i], pos_label=1)
                #     plt.plot(fpr, tpr, 'k-', alpha=0.5,
                #              label='{:.1e} (shots)'.format(self.shots_values[i]))
                # else:
                #     fpr, tpr, thres = roc_curve(self.y_true, self.y_pred_shots[i], pos_label=1)
                #     plt.plot(fpr, tpr, 'k-', alpha=0.5)
        else:
            fpr, tpr, thres = roc_curve(self.y_true, self.y_pred, pos_label=1)
            plt.plot(fpr, tpr, linestyle='-', color='blue', label='{}'.format(self.backend_name))

        if self.backend_name != "ibmq_qasm_simulator":
            fpr_sim, tpr_sim, thres_sim = roc_curve(self.y_true, self.y_simulator_pred, pos_label=1)
            plt.plot(fpr_sim, tpr_sim, linestyle='--', color='black', label='{}'.format("ibmq_qasm_simulator"))

        plt.title('ROC Curve ({})'.format(self.backend_name))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig('ROC', dpi=300)
        plt.show()

    def generate_dataset(self):
        # generate 2d classification dataset
        X, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=[0.1, 0.1], center_box=(0.25, 0.75))
        # scatter plot, dots colored by class value
        df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        colors = {0: 'red', 1: 'blue'}
        fig, ax = pyplot.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        pyplot.xlim(0, 1)
        pyplot.ylim(0, 1)
        pyplot.show()

        df.to_csv("make_blobs_dataset.csv")

    def run_circuit_from_qasm(self, qasm_file, backend, shots=100000):
        backend = self.provider.get_backend(backend)
        qasm_circuit = QuantumCircuit.from_qasm_file('./{}'.format(qasm_file))
        qasm_circuit.draw(output='mpl')
        plt.show()
        job = execute(
            transpile(qasm_circuit, backend),
            backend=backend,
            shots=shots
        )
        result = job.result()
        print(result)

    def remove_idle_qwires(self, circ):
        dag = circuit_to_dag(circ)

        idle_wires = list(dag.idle_wires())
        for w in idle_wires:
            dag._remove_idle_wire(w)
            dag.qubits.remove(w)

        dag.qregs = OrderedDict()

        return dag_to_circuit(dag)
