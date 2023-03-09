import subprocess
import argparse

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg1', type=str, default=None)
    parser.add_argument('cfg2', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = options()
    assert args.cfg1
    p1 = subprocess.Popen(['python', 'train.py', '--c', args.cfg1])
    p2 = None
    if args.cfg2:
        p2 = subprocess.Popen(['python', 'train_others_gpu.py', '--c', args.cfg2])
    p1.wait()
    if p2:
        p2.wait()