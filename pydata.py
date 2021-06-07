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
# lists are mutable
a_list = ['foo',2,[4,5]]
a_list[2] = (3,4)
a_list
# tuples are immutable
a_tuple = (3,5,(4,5))
# a_tuple[1] = 'four'

