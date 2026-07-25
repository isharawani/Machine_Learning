import numpy as np

#arr = np.array([1, 2, 3])
#print (type(arr))
#print(arr * 2)
#print(arr.dtype)
#print (arr**2)

a=np.array([[1,2,3],[4,5,6]])
print(a.ndim)
print(a.shape[1])
print(a[1,0])

a=np.arange(1,20,2)
print(a)
print(list(range(10)))

b=np.random.permutation(np.arange(10))  #it create & randomly swap the numbers 
print(b)
c=np.random.rand(10)  #it create 10 random numbers between 0 and 1
print(c)

d=np.random.rand(2,3,4,2)  # 2 denots block, 3 denots 3 row in each block, 4 denots that each row has column 4, 2 denots each element has 2 numbers
print(d)

e=np.zeros((2,3))  #it create a 2x3 array with all elements as 0
print(e)


#slicing
f=np.array([1,2,3])
b=f[0:2]
print(b)

g=np.array([[1,2,3],[4,5,6]])
print(g[0:2,1:3])#0:2--takes rows 0 and 1 , 1:3--takes columns 1 and 2

a=np.arange(100)
print(a[1:5])
print(a[1:20:2])
print(a[ : : -5])
