from qml_inference_circuit import *
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from numpy import where, array
import csv
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

from tqdm import tqdm


class model_testing:
    def __init__(self, trained_weights=None, inner_product_bit_accuracy=None, activation_fxn_bit_accuracy=None,
                 debug=False):
        self.debug = debug
        self.beta = trained_weights
        self.ip_bit_accuracy = inner_product_bit_accuracy
        self.fxn_bit_accuracy = activation_fxn_bit_accuracy
        self.scaling_factor = 7
        self.f = lambda z: 1 / (1 + exp(-1 * self.scaling_factor * z))

        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='strangeworks-hub', group="science-team", project="science-test")

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
        self.y_pred = []
        self.y_simulator_pred = []

    def run_model_on_backend(self, backend, shots=50000):
        self.backend_name = backend
        self.backend = self.provider.get_backend(backend)

        circuit_batch_list = []
        for i in range(len(self.x_test)):
            if i % 2 == 0:
                print("Building circuit(s) ... {} %".format(round(100 * i / len(self.x_test), 3)))
            self.circuit = basis_encoding_circuit()

            self.circuit.encode_data(self.x_test[i])
            self.circuit.add_inner_product_module(self.beta, bit_accuracy=self.ip_bit_accuracy)
            self.circuit.add_activation_fxn_module(self.f, bit_accuracy=self.fxn_bit_accuracy)

            if self.debug:
                print("TRUE DOT: ", dot(self.x_test[i], self.beta))
                print("TRUE SIGMOID: ", self.f(dot(self.x_test[i], self.beta)))
                self.circuit.draw_circuit()
                # self.circuit.inference_circuit.qasm(filename='qasm')

            if self.circuit.inference_circuit.num_qubits <= 10:
                circuit_batch_list.append(self.circuit.inference_circuit)
            else:
                print(self.x_test[i])
        print("Building circuit(s) ... Done \n")

        print("Running circuit(s) on {} ... ".format(self.backend_name))
        job = execute(
            transpile(circuit_batch_list, self.backend),
            backend=self.backend,
            shots=shots
        )
        self.result = job.result()
        print("Done. \n")

        if self.backend_name != "ibmq_qasm_simulator":
            print("Running circuit(s) on ibmq_qasm_simulator ... ")
            self.sim_backend = self.provider.get_backend("ibmq_qasm_simulator")
            job = execute(
                transpile(circuit_batch_list, self.sim_backend),
                backend=self.sim_backend,
                shots=shots
            )
            self.simulation_results = job.result()
            print("Done. \n")

            for i in range(len(circuit_batch_list)):
                counts = self.simulation_results.get_counts(i)
                y_simulation_pred_value = self.get_class(counts)
                self.y_simulator_pred.append([y_simulation_pred_value])

        for i in range(len(circuit_batch_list)):
            counts = self.result.get_counts(i)
            y_pred_value = self.get_class(counts)
            self.y_pred.append([y_pred_value])

    def print_metrics(self):
        print("Printing metrics for backend {} ... \n".format(self.backend_name))

        self.print_confusion_matrix_and_measures()
        self.print_auc_score()
        self.plot_roc_curve()

    def print_backend_info(self):
        plot_gate_map(self.backend)
        plot_error_map(self.backend)
        # plot_coupling_map(self.backend)
        plt.show()

    def get_class(self, counts):
        total_shots = sum(list(counts.values()))
        for key, value in counts.items():
            if key == "1":
                y_pred = where(value / total_shots >= .5, 1, 0)
        return y_pred

    def print_confusion_matrix_and_measures(self):
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

    def print_auc_score(self):
        auc_score = roc_auc_score(self.y_true, self.y_pred)
        print("AUC = ", auc_score)

    def plot_roc_curve(self):
        fpr, tpr, thres = roc_curve(self.y_true, self.y_pred, pos_label=1)
        plt.plot(fpr, tpr, linestyle='-', color='blue', label='{}'.format(self.backend_name))

        if self.backend_name != "ibmq_qasm_simulator":
            fpr_sim, tpr_sim, thres_sim = roc_curve(self.y_true, self.y_simulator_pred, pos_label=1)
            plt.plot(fpr_sim, tpr_sim, linestyle='--', color='black', label='{}'.format("ibmq_qasm_simulator"))

        plt.title('ROC curve for Logistic Regression')
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

    def run_circuit_from_qasm(self, qasm_file, backend, shots=32000):
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
