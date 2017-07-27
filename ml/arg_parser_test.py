import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--square", help="display a square of a given number", type=int)
parser.add_argument("--data", help="data 2 ", type=float, default=1.2)

args = parser.parse_args()

print args.square**2
print args.data*2

"""
awp4211:Desktop xiyou$ python test.py --square 2
4
2.4

awp4211:Desktop xiyou$ python test.py
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    print args.square**2
TypeError: unsupported operand type(s) for ** or pow(): 'NoneType' and 'int'

awp4211:Desktop xiyou$ python test.py --square 2 --data 1.3
4
2.6
awp4211:Desktop xiyou$ 

"""