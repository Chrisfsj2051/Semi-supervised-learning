import subprocess
import argparse
import time

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgs', type=str, default=None, nargs='+')
    return parser.parse_args()

if __name__ == '__main__':
    args = options()
    p_list = []
    for cfg in args.cfgs:
        p = subprocess.Popen(['python', 'train_ori.py', '--c', cfg])
        p_list.append(p)
        time.sleep(10)
    print(p_list)
    while True:
        for i, p in enumerate(p_list):
            poll = p.poll()
            if poll is not None and poll != 0:
                print('Restart')
                p_list[i] = subprocess.Popen(['python', 'train_ori.py', '--c', args.cfgs[i]])
            time.sleep(10)
