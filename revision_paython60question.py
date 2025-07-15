#PANDAS
#ANS1
import pandas as pd
s=pd.read_csv('sales_data project 1.csv',encoding='latin1')
print(s.head())
#Ans2
import pandas as pd
s={
    'Name':['Mayank',None,'Preeti'],
    'Grade':['A',None,'A++']
}
p=pd.DataFrame(s)
print(p)
print(p.isnull().sum())
#Ans3
import pandas as pd
q={
    'Mark1':[99,None,99],
    'Mark2':[88,None,96],
    'Mark3':[95,None,88]
}
t=pd.DataFrame(q)
fill=t.fillna(t.mean(numeric_only=True))  #numeric_only=True means only int,float 
print('Filled the null values by the mean:=')
print(fill)
#Ans4
import pandas as pd
sd={
    'Name':['Mayank','Dev','Sourabh','Mukesh','Mayank','Dev','Mukesh'],
    'Marks1':[99,99,78,98,99,94,78],
    'Marks2':[88,91,99,95,99,88,81]
}
ds=pd.DataFrame(sd)
print(ds)
if 'Name' in ds.columns:
    print(ds.groupby('Name').mean())
else:
    print('Colomn Name not in DataFrame')
#Ans6
import pandas as pd
it={
    'Name':['Mayank','Dev','Rohan'],
    'Age':[28,26,24]
}
h=pd.DataFrame(it)
print(h[h['Age']>25])
#Ans7
import pandas as pd
it={
    'Name':['Mayank','Dev','Rohan'],
    'Age':[28,26,24],
    'Salary':[28000,15000,45000]
}
h=pd.DataFrame(it)
print(h.sort_values('Salary'))
#Ans8
import pandas as pd
it={
    'Name':['Mayank','Dev','Rohan'],
    'Age':[28,26,24],
    'Salary':[28000,15000,45000]
}
h=pd.DataFrame(it)
a=h.rename(columns={'Salary':'Emp_Salary'})
print(a)
#Ans9
import pandas as pd
df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})
df2 = pd.DataFrame({
    'id': [2, 3, 4],
    'score': [85, 90, 95]
})
k=pd.merge(df1,df2,on='id')
print(k)
#Ans10
import pandas as pd
df = pd.DataFrame({
    'region': ['East', 'West', 'East', 'South', 'West', 'East'],
    'sales': [100, 150, 200, 130, 170, 120]
})
f=pd.pivot_table(df,index='region',values='sales',aggfunc='sum')
print(f)
#NUMPY
#Ans11
import numpy as np 
s=np.arange(1,11,1)
print(s)
#Ans12
import numpy as np
s=np.array([1,2,3,4,5,6,7,8,9])
a=s.reshape([3,3])
print(a)
#Ans13
n=np.eye(2,2)
print(n)
#Ans14
s=np.array([1,2,3,4,5,6,7,8,9])
t=np.array([9,8,7,6,5,4,3,2,1])
d=s+t
print(d)
#Ans15
k=np.mean(s)
p=np.std(s)
print(k)
print(p)
#Ans16
arr = np.array([5, -3, 7, -1, 0, 4, -6])
arr[arr<0] = 0
print(arr)
#Ans17
ad=np.random.rand(10)
print(ad)
#Ans18
e=np.array([1,2,3,4,5,6,7,8,9])
print(e[e>5])
#Ans19
e=np.array([1,2,3,4,5,6,7,8,9])
print(np.vstack(e))
print(np.hstack(e))
#Ans20
arr = np.array([[1, 2, 3], [4, 5, 6]])
print('Flatten =',arr.flatten())
print('Reval =',arr.ravel())
print('Reshape =',arr.reshape(6,1))
#Matplotlib 
#Ans21
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[2,4,6,8,10]
plt.plot(x,y,marker='x',linestyle='dotted',color='red',label='PLOTING')
plt.title('LINE GRAPH')
plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')
plt.grid()
plt.legend()
plt.show()
#Ans22
import matplotlib.pyplot as plt
sales=[200,1001,1900]
product=['kurkure','chips','popcorn']
plt.bar(product,sales,color=['red','green','yellow'],label=['kurkure','chips','popcorn'])
plt.title('BAR GRAPH')
plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')
plt.grid()
plt.legend()
plt.show()
#Ans23
import matplotlib.pyplot as plt
import numpy as np
x=np.random.randn(100)
plt.hist(x,bins=30,color='lightgreen',edgecolor='black')
plt.show()
#Ans24
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[2,4,6,8,10]
plt.scatter(x,y,marker='x',color='red')
plt.title('SCATTER PLOT')
plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')
plt.show()
#Ans25
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[2,4,6,8,10]
a=[1,2,3,4,5]
b=[2,4,6,8,10]
plt.plot(x,y,marker='x',linestyle='dotted',color='red',label='PLOTING1')
plt.plot(b,a,marker='s',linestyle='dotted',color='green',label='PLOTING2')
plt.title('LINE GRAPH')
plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')
plt.grid()
plt.legend()
plt.show()
#Ans26
import matplotlib.pyplot as plt
sales=[200,1001,1900]
product=['kurkure','chips','popcorn']
plt.pie(sales,labels=product,autopct='%1.1f%%',colors=['red','green','yellow'])
plt.legend()
plt.tight_layout()
plt.show()
#Ans27
import matplotlib.pyplot as plt
mark1=[100,89,76]
mark2=[99,78,89]
mark3=[88,77,99]
data=[mark1,mark2,mark3]
p=plt.boxplot(data,patch_artist=True,labels=['mark1','mark2','mark3'])
colors=['red','coral','teal']
for patch,color in zip(p['boxes'],colors):
    patch.set_facecolor(color)
plt.show()
#Ans28
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[2,4,6,8,10]
a=[1,2,3,4,5]
b=[2,4,6,8,10]
plt.plot(x,y,marker='x',linestyle='dotted',color='red',label='PLOTING1')
plt.plot(b,a,marker='s',linestyle='dotted',color='green',label='PLOTING2')
plt.title('LINE GRAPH')
plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')
plt.grid()
plt.legend()
plt.show()
#Ans29
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[2,4,6,8,10]
a=[1,2,3,4,5]
b=[2,4,6,8,10]
plt.plot(x,y,marker='x',linestyle='dotted',color='red',label='PLOTING1')
plt.plot(b,a,marker='s',linestyle='dotted',color='green',label='PLOTING2')
plt.title('LINE GRAPH')
plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')
plt.grid()
plt.legend()

plt.savefig('savekrlo.png',dpi=1200,bbox_inches='tight')
plt.show()
#Ans30
import matplotlib.pyplot as plt
import numpy as np
data1=np.array([[77,88,99],
               [87,88,67],
               [88,76,89]])
plt.imshow(data1,cmap='RdYlGn_r',interpolation='nearest')
plt.colorbar(label='MARKS')
plt.title('HEAT MAP')
plt.xlabel('X-AXIS')
plt.ylabel('Y-AXIS')
plt.xticks([0,1,2],['Mayank','Dev','Sonu'])
plt.yticks([0,1,2],['English','Science','Maths'])
plt.show()
#FUNCTIONS
#Ans31
def sq(a):
    return a**2
print(sq(4))
#Ans32
def p(n):
    
    if n[::-1]==n:
        print(f'Yes {n} is the polindrome')
    else:
        print('NO Its not')
p(input('Enter any string to check Polindrome='))
#Ans33
def count_vowels(s):
    count=0
    for i in s.lower():
        if i in 'aeiou':
            count+=1
    return count
p=input('Enter any name to count vowels=')
print("Number of vowels:", count_vowels(p))
#Ans34
def fact(n):
    fact1=1
    for i in range(1,n+1):
        fact1*=i
    return fact1
p=int(input('Enter any number to get factorial='))
print(fact(p))
#Ans35
n=int(input('Enter any number to get the sum ='))
def add(a):
    d=0
    for i in range(1,a):
        digit=a%10
        d+=digit
        a=a//10
    return d
print(add(n))
#OOPS
# Ans36
class constructor():
    def __init__(self,name,english,hindi):
        self.name=name
        self.english=english
        self.hindi=hindi
    def got(self):
        print(f'I got {self.english} and {self.hindi} and My Name is {self.name}')
d=constructor(name='Mayank',english=98,hindi=92)
d.got()
#Ans37
class animal():
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def info(self):
        print(f'My Pet Name is {self.name} and age of my pet is {self.age}')
d=animal('Bruno',10)
d.info()
class dog(animal):
    def __init__(self, name, age,variety):
        super().__init__(name, age)
        self.variety=variety
    def breed(self):
        print(f'My Pet Name is {self.name} and age of my pet is {self.age} and breed is {self.variety}')
d1=dog('Tommy',11,'Labera')
d1.breed()
#Ans38
class method():
    student_name='Rao Man Singh'
    def __init__(self,name,grade):
        self.name=name
        self.grade=grade
    def info(self):
        print(f'My Name is {self.name} and my grade is {self.grade}')
k=method('Mayank','A')
k.info() #class- method
print(method.student_name) #static method
#Ans39
class point:
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __add__(self,other):
         return point(self.x + other.x,self.y+other.y)
    def display(self):
        print(f'{self.x}:{self.y}')
p1=point(4,5)
p2=point(3,2)
p3=p1+p2
p3.display()
#Ans40
class animal():
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def info(self):
        print(f'My Pet Name is {self.name} and age of my pet is {self.age}')
d=animal('Bruno',10)
d.info()
class dog(animal):
    def __init__(self, name, age,variety):
        super().__init__(name, age)
        self.variety=variety
    def breed(self):
        print(f'My Pet Name is {self.name} and age of my pet is {self.age} and breed is {self.variety}')
d1=dog('Tommy',11,'Labera')
d1.breed()

# 41. Write a loop to print even numbers from 1 to 20.
for i in range(1, 21):
    if i % 2 == 0:
        print(i)

# 42. Reverse a string without using slicing.
s = "hello"
reversed_str = ""
for char in s:
    reversed_str = char + reversed_str
print(reversed_str)

# 43. Count the frequency of each word in a string using a dictionary.
text = "hello world hello again"
words = text.split()
freq = {}
for word in words:
    freq[word] = freq.get(word, 0) + 1
print(freq)

# 44. Add an element to a set and check membership.
s = {1, 2, 3}
s.add(4)
print(4 in s)

# 45. Convert a list into a tuple and vice versa.
lst = [1, 2, 3]
tpl = tuple(lst)
lst2 = list(tpl)
print(tpl)
print(lst2)

# 46. Create a dictionary with student names and their marks.
students = {"Raj": 85, "Priya": 92, "Amit": 78}
print(students)

# 47. Find the max, min, and sum of a list of numbers.
nums = [4, 7, 1, 9, 3]
print(max(nums))
print(min(nums))
print(sum(nums))

# 48. Use a for loop to print each key-value pair from a dictionary.
for k, v in students.items():
    print(k, v)

# 49. Merge two lists into a dictionary using zip().
keys = ["a", "b", "c"]
values = [1, 2, 3]
merged_dict = dict(zip(keys, values))
print(merged_dict)

# 50. Demonstrate the use of break and continue in a loop.
for i in range(1, 10):
    if i == 5:
        continue
    if i == 8:
        break
    print(i)
