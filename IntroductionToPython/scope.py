x = 2
def f():
    x = 3
    return x
print(x)
print(f())

x = 5
def f():
    y = 2*x
    return y
print(f(y))

import builtins
dir(builtins)