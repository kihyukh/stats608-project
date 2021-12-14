import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

x = np.linspace(0, 1, 100)
rv1 = beta(10, 8)
rv2 = beta(1.5, 2)
plt.plot(x, rv1.pdf(x), 'k-', lw=2, label='action 1')
plt.plot(x, rv2.pdf(x), 'b-', lw=2, label='action 2')
plt.legend(loc='best', frameon=False)
plt.show()