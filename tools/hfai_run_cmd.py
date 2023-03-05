import argparse
import os

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_folder', type=str)
    parser.add_argument('--keyword', default='', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_config()
    path = args.cfg_folder
    hfai_formatter = 'hfai python train.py --c %s/%s -- --nodes 1 --name %s &'

    for root, _, filenames in os.walk(args.cfg_folder):
        for filename in filenames:
            if not filename.endswith('yaml') or 'archive' in root:
                continue
            if args.keyword and args.keyword not in filename:
                continue
            print(hfai_formatter % (root, filename, filename[:-5]))
