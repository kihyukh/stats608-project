import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import json


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--log')
parser.add_argument('--t')

args = parser.parse_args()

filename = args.log

tt = [int(a) for a in args.t.split(',')]
fig, axs = plt.subplots(len(tt))
histogram = []
figure(figsize=(6, 1.5), dpi=80)
plot_count = 0
with open(filename, 'r') as f:
    for line in f:
        t, a, y, r, h = line.rstrip().split('\t', 4)
        if int(t) not in tt:
            continue
        histogram = json.loads(h)
        histogram = [0.] * 200 + histogram
        if len(histogram) < 1000:
            last = histogram[-1] + histogram[-2] # oops off by one error when using tau. Thankfully, does not hurt the algorithm.
            avg = last / (1000 - len(histogram) + 2)
            histogram[-1] = avg
            histogram[-2] = avg
            histogram = histogram + [avg] * (1000 - len(histogram))
        axs[plot_count].bar(
            x=list(range(len(histogram))), height=histogram, width=1,
            label='haha'
        )
        axs[plot_count].set_ylim([0, 0.1])
        axs[plot_count].set_ylabel('probability')
        axs[plot_count].set_title('Posterior of tau at t = {}'.format(t))
        axs[plot_count].axvline(x=500, color='r', linestyle='--')
        plot_count += 1
fig.tight_layout(rect=[0, 0.06, 1, 0.85])
plt.show()
