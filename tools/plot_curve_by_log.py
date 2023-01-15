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
        x = [item['iteration'] for item in logs]
        y = [item[args.metric] for item in logs]
        plt.plot(x, y, label=log)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()