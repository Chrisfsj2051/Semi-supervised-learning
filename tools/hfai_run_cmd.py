import argparse
import os

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_folder', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_config()
    path = args.cfg_folder
    hfai_formatter = 'hfai python train.py --c %s/%s -- --nodes 1 --name %s &'
    for filename in os.listdir(path):
        if not filename.endswith('yaml'):
            continue
        print(hfai_formatter % (path, filename, filename[:-5]))
