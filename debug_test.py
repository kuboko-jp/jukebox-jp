import sys

def hello(args):
    a = 0
    b = 1
    c = a
    print(f"hello args : {args}")

if __name__ == '__main__':
    hello(sys.argv)