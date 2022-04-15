import matplotlib.pyplot as plt
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))

z_evals = np.arange(-7, 7, 0.1)

plt.plot(z_evals, sigmoid(z_evals), 'k', linewidth=1.5)
plt.title("Sigmoid Activation Function")
plt.xlabel(r'$z$')
plt.ylabel(r'$\sigma(z)$')
plt.savefig(fname='sigmoid_activation_fxn', dpi=300)
plt.show()
