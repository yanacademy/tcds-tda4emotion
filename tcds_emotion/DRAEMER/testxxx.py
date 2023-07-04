import sys
def prime(x, n):
    if x==1:
        return False
    if n<x:
        if x%n == 0:
            return False
        else:
            return prime(x,n+1)
    else:
        return True


for i in range(1,1000):
    if prime(i,2):
        print(i)

