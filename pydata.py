# Python for Data Analysis
import numpy as np
import os
# create a python dictionary with 10 random numbers, indexed from 0 to 9
data={i: np.random.randn() for i in range(10)}
data
# the current_path function
def current_path():
    """
    Return the current working directory path.
    Examples
    --------
    In [9]: current_path()
    Out[9]: Current working directory: C:/Users/Dave
    """
    try:
      return print("Current working directory:", os.getcwd())
    except FileNotFoundError:
      raise UsageError("CWD no longer exists - please use os.chdir() to change directory.")  

os.chdir('C:/Users')
current_path()
# create a list of integers and show that b=a sets b as a reference to a
a = [1,2,3]
b = a
a.append(4)
print(b)
# object references have no type associated with them
a = 5
type(a)
a = 'foo'
type(a)
# using the is keyword to test if two references refer to the same object
a = [1,2,3]
b = a 
c = list(a)
a is b
# the list functions creates a distinct copy 
c is not a
a == c
# None for a python variable is similar to NULL for a database variable
a = None
a is None # True
#you need an extra backslash characer if the text requires one
date_today='2021\\06\\12'
print(date_today)
#the r prefixed to the first quote tells Python not to treat any characters as special characters
date_yesterday=r'2021\06\11'
print(date_yesterday)
#lists, dicts, NumPy arrays are mutable meaning the can be modified 
a_list = ['foo', 2, [4, 5]]
a_list[2] = (3, 4)
#strings and tuples are immutable meaning they cannot be modified after they are defined
a_tuple = (3, 5, (4, 5))
#a_tuple[1] = 'four' # causes an error message
#you can use either triple or single quotes for strings that span multiple lines
s="""
This string spans 
multiple lines so use triple quotes
"""
s.count('\n') # 3 - the number of new lines in s
#strings are unmutable, you cannot change a string, but you can modify a second variable based on the first variable
b=a.replace('string','longer string')
#converting datatypes
a = 5.6
s = str(a)
print(s)
# strings can be treated like lists
s='python'
list(s) #{'p','y','t','h','o','n']
s[:3] #pyt
# adding strings together concatenates the strings
a = 'this is the first half '
b = 'and this is the second half'
a + b
# string objects can be formatted by using format method
#{0:.2f} - format the first argument as a floating-point number with 2 decimal places
#{1:s} - format the second argument to the format function as a string
#{2:d} - means to format the third argument as an exact integer
template = '{0:.2f} {1:s} are worth US${2:d}' 
template.format(1.22, 'Canadian Dollars', 1)
# Type casting - changing the data types to and from string, integer and float
s = '3.14159' #string pi
fval = float(s) #float pi
type(fval) #float
int(fval) #removes the decimals so we are left with 3
bool(fval) #True - any non-zero number is considered as True in boolean expressions
bool(0) #False
# None is the Python null value type
a = None
a is None
b = 5
b is not None
# None is a common default value for function arguments
def add_and_maybe_multiply(a, b, c=None):
    result = a + b
    if c is not None:
        result = result * c
    return result
# None is a reserved keyword and it has it's own type called NoneType
type(None) #NoneType
#the python built-in datetime module provides datetime, date and time types
import datetime
from datetime import datetime, date, time
dt = datetime(2011, 10, 29, 20, 30, 21)
dt.day #29
dt.minute #30
dt.date() #datetime.date(2011, 10, 29)
dt.time() # datetime.time(20, 30, 21)
#the strftime function formats a datetime as a string
dt.strftime('%m/%d/%Y %H:%M')
#turning a string into a datetime object with the strptime function
datetime.strptime('20091031', '%Y%m%d')
#getting rid of the minutes and seconds in the datetime object
dt.replace(minute=0, second=0)
dt2 = datetime(2011, 11, 15, 22, 30)
# calculating the days and seconds between dates
delta = dt2 - dt
print(delta)
type(delta)
dt
# adding a timedelta to a datetime produces a new shifted datetime
dt + delta
# the if statement executes the code block that follows the statement that evaluates to True
x=1
if x < 0:
    print("It's negative")
elif x == 0:
    print('Equal to zero')
elif 0 < x < 5:
    print('Positive but smaller than 5')
else:
    print('Positive and larger than or equal to 5')
# a compound condition using the or operator
a = 5; b = 7
c = 8; d = 4
if a < b or c > d:
    print('Made it')
# chaining comparisons is allowed in python
if 4 > 3 > 2 > 1:
    print('chained comparisons work!')
# for loops iterate over a collection (list or tuple) or an iterator
sequence = [1, 2, None, 4, None, 5]
total = 0
for value in sequence:
    if value is None:
        continue # used to avoid an error from summing over None
    total += value
    
print(total)
# a for loop can be exited altogether with the break keyword
sequence = [1, 2, 0, 4, 6, 5, 2, 1]
total_until_5 = 0
for value in sequence:
    if value == 5:
        break
    total_until_5 += value

total_until_5
# the break keyword only terminates the innermost for loop
for i in range(4):
    for j in range(4):
        if j > i:
            break
        print((i,j))
# a while loop specifies a condition and a block of code to be run until the condition evaluates to false
x = 256
total = 0
while x > 0:
    if total > 500:
        break
    total += x
    x = x // 2
    print(x)

print(total)

# the pass keyword means to do nothing, it's needed because Python uses whiespace to delimit blocks
if x < 0:
    print('negative!')
elif x == 0:
    # To Do: put something smart here when we figure out what x==0 means
    pass
else:
    print('positive!')

#the range function returns an iterator
range(10)
list(range(10))
# the arguments to the range function are start, end and step. The end point is not included in the list.
list(range(0, 20, 2)) #[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
list(range(5, 0, -1)) #[5, 4, 3, 2, 1]
# a common use of a range object is to iterate over it in a for loop
seq = [1, 2, 3, 4]
for i in range(len(seq)):
    val = seq[i]
    print(val)
# sum up all of the numbers from 0 to 99,999 that are multiples of 3 or 5
sum = 0
for i in range(100000):
    # % is the modulo operator
    if i % 3 == 0 or i % 5 == 0:
        sum += i

print(sum)
# ternary expressions have the same effect as the usual if condition: value if true else: value if false
x = -2
'Non-negative' if x >= 0 else 'Negative'


a is b
# the list functions creates a distinct copy 
c is not a
a == c
# lists are mutable
a_list = ['foo',2,[4,5]]
a_list[2] = (3,4)
a_list
# tuples are immutable
a_tuple = (3,5,(4,5))
# a_tuple[1] = 'four'

