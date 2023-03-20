import subprocess
import argparse
import time
import os

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str, default=None, nargs='+')
    parser.add_argument('--gpus', type=int, default=None, nargs='+')
    parser.add_argument('--gpu_start', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = options()
    p_list = []
    if args.gpus == None:
        assert 8 % len(args.cfgs) == 0
        args.gpus = [8 // len(args.cfgs)]
    gpu_st = args.gpu_start
    for gpu, cfg in zip(args.gpus, args.cfgs):
        cur_gpus = ''.join([str(int(gpu_st + i)) + ',' for i in range(gpu)])
        cur_gpus = cur_gpus[:-1]
        gpu_st += gpu
        new_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cur_gpus)}
        p = subprocess.Popen(['python', 'train_ori.py', '--c', cfg],
                             env=new_env)
        p_list.append(p)
        time.sleep(5)

    while True:
        finished_cnt = 0
        for i, p in enumerate(p_list):
            poll = p.poll()
            if poll is not None and poll != 0:
                print('Restart')
            elif poll == 0:
                finished_cnt += 1
            time.sleep(5)
        if finished_cnt == len(p_list):
            break
