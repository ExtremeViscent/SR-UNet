import os
import sys
from glob import glob


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: edit_config.py <file_reg> <start_with> <value>')
        sys.exit(1)
    list_file = glob(sys.argv[1])
    for file in list_file:
        print(file)
        with open(file, 'r') as f:
            lines = f.readlines()
        with open(file, 'w') as f:
            for line in lines:
                if line.startswith(sys.argv[2]):
                    line = sys.argv[3] + '\n'
                    print('Changed:', line)
                f.write(line)
    print('Done!')
    sys.exit(0)
