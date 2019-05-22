dictionary = {'spain': 'madrid', 'usa' : 'vegas'}
#print(dictionary.keys())
#print(dictionary.values())

dictionary['spain'] = "barcelona"
print(dictionary)
dictionary['france'] = "paris"
print(dictionary)
del dictionary['spain']
print(dictionary)
print('france' in dictionary)
dictionary.clear()
print(dictionary)