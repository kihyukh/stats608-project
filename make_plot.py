import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--log')
parser.add_argument('--label')

args = parser.parse_args()

filename = args.log
labels = args.label

if labels:
    labels = labels.split(',')
figure(figsize=(6, 2), dpi=80)
for i, filename in enumerate(args.log.split(',')):
    ts = []
    regret = []
    with open(filename, 'r') as f:
        for line in f:
            t, a, y, r = line.rstrip().split('\t')
            ts.append(int(t))
            regret.append(float(r) / int(t))
    sns.lineplot(x=ts, y=regret, label=labels[i] if labels else '')

plt.xlabel('time')
plt.ylabel('average regret')
plt.show()
