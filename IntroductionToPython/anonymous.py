
## map
number_list = [1,2,3]
y = map(lambda x:x**2, number_list)
print(list(y))
## iterators

name = "ronaldo"
it = iter(name)
print(next(it))
print(*it)

##zip

list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)

un_zip = zip(*z_list)
un_list1, un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))

num1 = [1,2,3]
num2 = [i + 1 for i in num1]
print(num2)

num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i + 5 for i in num1]
print (num2)