import argparse
import re
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=str, required=True, nargs='+')
    parser.add_argument('--metric', type=str, required=True)
    return parser.parse_args()

def analysis_log(log_path):
    logs = []
    with open(log_path, 'r') as f:
        contents = f.readlines()
    contents = [line.strip() for line in contents if 'BEST' in line]
    for line in contents:
        line = line[31:]
        it = int(line.split(' ')[0])
        metrics = re.findall('{.*}', line)[0]
        metrics = metrics.replace("'", '"')
        metrics = json.loads(metrics)
        metrics['iteration'] = it
        logs.append(metrics)
    return logs

def main():
    args = parse_args()
    for log in args.logs:
        logs = analysis_log(log)
        if args.metric not in logs[0]:
            continue
        x = [item['iteration'] for item in logs]
        y = [item[args.metric] for item in logs]
        tmp_x, tmp_y = [], []
        prev_max_iter = 1e9
        for i in reversed(range(len(x))):
            if x[i] < prev_max_iter:
                prev_max_iter = x[i]
                tmp_x.append(x[i])
                tmp_y.append(y[i])
        x = list(reversed(tmp_x))
        y = list(reversed(tmp_y))
        plt.plot(x, y, label=log)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel(args.metric)
    plt.show()

if __name__ == '__main__':
    main()