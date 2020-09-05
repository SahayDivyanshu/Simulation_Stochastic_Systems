import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

n = 5
i = 0
while 1:
  i = ncr(119,n)/ncr(125,n)
  n = n + 1
  if (i <= 0.05):
      break
print('----value of n----')
print(n)
print('---- rejection ---')
print(1-i)
print('---- acceptance ---')
print(i)
