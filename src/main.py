import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('directory', help='Path to the input file')

args = parser.parse_args()

if __name__ == "__main__":
    print(args.directory)

    