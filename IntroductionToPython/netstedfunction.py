def square():
    def add():
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())

def f(a, b = 1, c = 2):
    y = a +b +c
    return y
print(f(5))
print(f(5,4,3))
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)
def f(**kwargs):
    for key ,value in kwargs.items():
        print(key, " ", value)
f(country = 'spain', capital = 'madrid', population = 123456)
