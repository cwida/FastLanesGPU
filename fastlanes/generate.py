from fls_gen.generate_bitpack_lib import *
from fls_gen.tools import *


def main():
    dir_path = "./generated/"

    creat_if_not_exist(dir_path)
    clear_prev_generation()
    generate_bitpack_lib()
    clang_format()


if __name__ == '__main__':
    main()
