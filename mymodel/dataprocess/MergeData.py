import os
import sys

if __name__ == '__main__':
    merge_file_path = "../../novel.txt"
    with open(merge_file_path, 'at') as writer:
        for filename in os.listdir('../novel'):
            print(filename)
            if not filename.endswith('.txt'):
                continue
            with open(os.path.join('../novel',filename), 'rt') as reader:
                line = reader.readline()
                writer.write(line)
    print("end")