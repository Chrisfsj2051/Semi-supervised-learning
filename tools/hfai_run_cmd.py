import argparse
import os

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_folder', type=str)
    parser.add_argument('--recursive', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_config()
    path = args.cfg_folder
    hfai_formatter = 'hfai python train.py --c %s/%s -- --nodes 1 --name %s &'

    if not args.recursive:
        for filename in os.listdir(args.cfg_folder):
            if not filename.endswith('yaml'):
                continue
            print(hfai_formatter % (path, filename, filename[:-5]))
    else:
        for root, _, filenames in os.walk(args.cfg_folder):
            for filename in filenames:
                if not filename.endswith('yaml') or 'archive' in root:
                    continue
                print(hfai_formatter % (root, filename, filename[:-5]))
