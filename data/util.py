import argparse
import os


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def gen_data(dataset, scale_factor):
    path = './' + dataset + '/dbgen/'
    with cd(path):
        os.system('rm -rf *.tbl')
        os.system('./dbgen -s %d -T a' % scale_factor)
        os.system('mkdir -p ../data/s%d' % scale_factor)
        os.system('mv *.tbl ../data/s%d/' % scale_factor)


def transform(dataset, scale_factor):
    path = './' + dataset + '/loader/'
    ip = '../data/s%d/' % scale_factor
    op = '../data/s%d_columnar/' % scale_factor
    with cd(path):
        os.system('mkdir -p %s' % op)
        os.system('python3 convert.py ../data/s%d/' % scale_factor)
        os.system(
            './loader --lineorder %s/lineorder.tbl --ddate %s/date.tbl --customer %s/customer.tbl.p --supplier %s/supplier.tbl.p --part %s/part.tbl.p --datadir %s' % (
                ip, ip, ip, ip, ip, op))


def sort(dataset, scale_factor):
    path = './' + dataset + '/loader/'
    ip = '../data/s%d/' % scale_factor
    op = '../data/s%d_columnar_sorted/' % scale_factor
    with cd(path):
        os.system('mkdir -p %s' % op)
        os.system('python3 convert.py ../data/s%d/' % scale_factor)
        os.system('python3 sort.py ../data/s%d/' % scale_factor)
        os.system(
            './loader --lineorder %s/lineorder.tbl.s --ddate %s/date.tbl --customer %s/customer.tbl.p --supplier %s/supplier.tbl.p --part %s/part.tbl.p --datadir %s' % (
                ip, ip, ip, ip, ip, op))


def sort_other_way(dataset, scale_factor):
    path = './' + dataset + '/loader/'
    ip = '../data/s%d/' % scale_factor
    op = '../data/s%d_columnar_sorted_other_way/' % scale_factor
    with cd(path):
        os.system('mkdir -p %s' % op)
        os.system('python3 convert.py ../data/s%d/' % scale_factor)
        os.system('python3 sort_other_way.py ../data/s%d/' % scale_factor)
        os.system(
            './loader --lineorder %s/lineorder.tbl.s --ddate %s/date.tbl --customer %s/customer.tbl.p --supplier %s/supplier.tbl.p --part %s/part.tbl.p --datadir %s' % (
                ip, ip, ip, ip, ip, op))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data gen')
    parser.add_argument('dataset', type=str, choices=['ssb'])
    parser.add_argument('scale_factor', type=int)
    parser.add_argument('action', type=str, choices=['gen', 'transform', 'sort', 'sort_other_way'])
    args = parser.parse_args()

    if args.action == 'gen':
        gen_data(args.dataset, args.scale_factor)
    elif args.action == 'transform':
        transform(args.dataset, args.scale_factor)
    elif args.action == 'sort':
        sort(args.dataset, args.scale_factor)
    elif args.action == 'sort_other_way':
        sort_other_way(args.dataset, args.scale_factor)
