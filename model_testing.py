from qml_inference_circuit import *
from sklearn.metrics import confusion_matrix

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
        self.f = lambda z: 1 / (1 + exp(-z))

        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='ibm-q')

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

    def run_model_on_backend(self, backend, shots=20000):

        for i in tqdm(range(len(self.x_test)), desc="Processing X Test Vectors"):
            if self.debug:
                print("TRUE DOT: ", dot(self.x_test[i], self.beta))
                print("TRUE SIGMOID: ", self.f(dot(self.x_test[i], self.beta)))
                self.circuit.draw_circuit()
                self.circuit.display_results()

            self.circuit = basis_encoding_circuit()
            self.circuit.encode_data(self.x_test[i])
            self.circuit.add_inner_product_module(self.beta, bit_accuracy=self.ip_bit_accuracy)
            self.circuit.add_activation_fxn_module(self.f, bit_accuracy=self.fxn_bit_accuracy)
            self.backend = self.provider.get_backend(backend)
            self.circuit.execute_circuit(shots=shots, backend=backend, optimization_level=None)

            counts = self.circuit.get_counts()
            y_pred_value = self.get_class(counts)

            self.y_pred.append([y_pred_value])

    def print_metrics(self):
        self.get_confusion_matrix()
        print("CONFUSION MATRIX: ", self.cfn_matrix)

    def get_class(self, counts):
        total_shots = sum(list(counts.values()))
        for key, value in counts.items():
            if key == "1":
                y_pred = where(value / total_shots >= .5, 1, 0)
        return y_pred

    def get_confusion_matrix(self):
        self.cfn_matrix = confusion_matrix(self.y_true, self.y_pred)

    def get_auc(self):
        pass

    def get_roc(self):
        pass

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
