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

os.chdir('C:/Users/Dave/Documents/Machine Learning/pydata-book') 
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
b=s.replace('string','longer string')
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
a + b #'this is the first half and this is the second half' 
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
sum_range = 0
for i in range(100000):
    # % is the modulo operator
    if i % 3 == 0 or i % 5 == 0:
        sum_range += i

print(sum_range)
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

# createing a tuple 
tup = 4, 5, 6
tup

# a tuple of tuples
nested_tup = (4, 5, 6), (7, 8)
nested_tup

# converting a sequence to a tuple
tuple([4, 0, 2])
# converting a string to a tuple
tup = tuple('string')
# elements are indexed starting at zero
tup[0]
tup[1]
# if an object inside a tuple is mutable, you can modify it in-place
tup = tuple(['foo', [1, 2], True])
tup[1].append(2)
tup
# concatenating tuples
(4, None, 'foo') + (6, 0) + ('bar',)
# multiplying tuples has the effect of concatenating together multiple copies of the tuple
('foo', 'bar') * 4
# unpacking a tuple like expression of variables
tup = (4, 5, 6)
a, b, c = tup
b
# even sequences with nested tuples can be unpacked
tup = 4, 5, (6, 7)
a, b, (c, d) = tup
d
# swapping variable names - a starts out as 1 but ends as 2
a, b = 1, 2
a
b
b, a = a, b
a
b
# a common use of unpacking is to iterate over a sequence of tuples
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c in seq:
    print('a={0}, b={1}, c={2}'.format(a, b, c)) # a is 1,4,7

# the *rest syntax can be used to pluck off the rest of the items
values = 1, 2, 3, 4, 5
a, b, *rest = values
a, b
rest # [3, 4, 5]
# the underscore can also be used instead of the rest keyword
a, b, *_ = values
_ # {3,4,5]
# cunt is one of the few tuple methods available
a = (1, 2, 2, 2, 3, 4, 2)
a.count(2) # 4
# defining a list using square brackets
a_list = [2, 3, 7, None]
tup = ('foo', 'bar', 'baz')
# transform a tuple into a list
b_list = list(tup)
b_list
# modifying the contents of a list in-place
b_list[1] = 'peekaboo'
b_list
# transofrming the range object into a list
gen = range(10)
gen
list(gen)
# adding elements to a list
b_list.append('dwarf')
b_list
# inserting a string into a specific location in the list
b_list.insert(1, 'red')
b_list
# pop is the inverse of insert as it allows you to delete an element at a location
b_list.pop(2)
b_list
# appending an element to the end of the list
b_list.append('foo')
b_list
# removing the first instance of foo
b_list.remove('foo')
b_list
# check if an item is in the list
'dwarf' in b_list 
# check if an item is not in the list
'dwarf' not in b_list 
# adding two lists together concatenates them
[4, None, 'foo'] + [7, 8, (2, 3)]
# appending multiple items with the extend method
x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])
x
# sorting a list in-place without creating a new object
a = [7, 2, 5, 1, 3]
a.sort()
a
# the sort key allows us to sort by the result of the len function
b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)
b
# import the built-in bisect module
import bisect
c = [1, 2, 2, 2, 3, 4, 7]
# the bisect method finds the locaiton where an element should be inserted to keep it sorted properly
bisect.bisect(c, 2) #4
bisect.bisect(c, 5) #6
# insort inserts an element so that the list remains sorted 
bisect.insort(c, 6) # [1, 2, 2, 2, 3, 4, 6, 7]
# select sections of most sequence types using slice notaion with the indexing operator []
seq = [7, 2, 3, 7, 5, 6, 0, 1]
# select from position 1 (the second element) up to, but not including, postion 5 (the sixth element)
seq[1:5] # [2, 3, 7, 5]
# insert a 6 and a 3 at position 4 and 5
seq[3:4] = [6, 3] # [7, 2, 3, 6, 3, 5, 6, 0, 1]
seq 
# go from the start up to but not including position 5 (the sixth element)
seq[:5] # [7, 2, 3, 6, 3]
# start from position 3 (the forth element) up to the end
seq[3:] # [6, 3, 5, 6, 0, 1]
# negative indicies slice starting from the end (going back four positions)
seq[-4:] # [5, 6, 0, 1]
seq[-6:-2] # [6, 3, 5, 6]
# taking every second item using the double colon operator ::
seq[::2]
# take every second item using the start:stop:step notation
seq[0:len(seq):2]
# reverse the sequence
seq[::-1]
# create a dictionary from a list with the enumerate function
some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, v in enumerate(some_list):
    mapping[v] = i

mapping # {'foo': 0, 'bar': 1, 'baz': 2}
# zip pairs up elements of sequences such as a list or a tuple and makes a list of tuples
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1, seq2)
list(zipped) #[('foo', 'one'), ('bar', 'two'), ('baz', 'three')]
# the number of elements of each tuple is determined by the shortest sequence
seq3 = [False, True]
list(zip(seq1, seq2, seq3)) #[('foo', 'one', False), ('bar', 'two', True)]
# using zip to iterate over multiple sequences
for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('{0}: {1}, {2}'.format(i, a, b))

#0: foo, one
#1: bar, two
#2: baz, three
# using zip to unzip a sequence 
pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'),
            ('Curt', 'Schilling')]
first_names, last_names = zip(*pitchers)
first_names #('Nolan', 'Roger', 'Curt') 
last_names # ('Ryan', 'Clemens', 'Schilling')
# using the reversed generator to iterate over a sequence in reverse order 
list(reversed(range(10)))
# dict's are also known as hash maps or associative arrays where the keys and values are python objects
empty_dict = {}
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
d1
# inserting an element
d1[7] = 'an integer'
d1 #{'a': 'some value', 'b': [1, 2, 3, 4], 7: 'an integer'}
# access an element by key
d1['b']
# check if the dict contains a key
'b' in d1
# insert another numeric key and a string value
d1[5] = 'some value'
d1 #
# insert a string key and a string value
d1['dummy'] = 'another value'
d1 #{'a': 'some value', 'b': [1, 2, 3, 4], 7: 'an integer', 5: 'some value', 'dummy': 'another value'}
# use the del keyword to delete the key-value pair when the key is 5
del d1[5]
d1 #{'a': 'some value', 'b': [1, 2, 3, 4], 7: 'an integer', 'dummy': 'another value'}
# use the pop method to delete the key and return it's value at the same time
ret = d1.pop('dummy')
ret
d1
# output the keys and values in the order that they exist in the dict
list(d1.keys())
list(d1.values())
# you can merge one dictionary to another with the update function
d1.update({'b' : 'foo', 'c' : 12})
d1 # {'a': 'some value', 'b': 'foo', 7: 'an integer', 'c': 12}
# creating a dict from two tuples
mapping = dict(zip(range(5), reversed(range(5))))
mapping #{0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
# categorizing a of words by thier first letters
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)

by_letter #{'a': ['apple', 'atom'], 'b': ['bat', 'bar', 'book']}
# check hashability, that is whether a Python object can be used as a key 
hash('string')
hash((1, 2, (2, 3)))
# hash((1, 2, [2, 3])) # throws an error message because lists are mutable
# in order to use a list as a key you must first convert it to a tuple
d = {}
d[tuple([1, 2, 3])] = 5
d # {(1, 2, 3): 5}
# a set is an unordered collection of unique elements that supports mathemtaical set operations: union, intersection, difference and symmetric difference
s=set([2, 2, 2, 1, 3, 3])
s # {1, 2, 3} 
# a set can be created via the set function or via curly braces
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
# the union is the set of distinct elements occuring in either set
a.union(b) #{1, 2, 3, 4, 5, 6, 7, 8}
# the pipe can be used instead of the union methond and | is known as bitwise OR
a | b # {1, 2, 3, 4, 5, 6, 7, 8}
# intersection contains the elements occurring in both sets
a.intersection(b) # {3, 4, 5}
# the ampersand is known as the bitwise and operator
a & b
# create a copy of a, same as setting c equal to a
c = a.copy()
c # {1, 2, 3, 4, 5}
# update c with values from b
c |= b
c # {1, 2, 3, 4, 5, 6, 7, 8}
# &= is the intersecion update operator - it sets d to be an intersection of a and b
d = a.copy()
d &= b
d # {3, 4, 5}
# similar to dicts, the elements of a set must be immutable, so you have to covert a list to a tuple before converting it to a set
my_data = [1, 2, 3, 4]
my_set = {tuple(my_data)}
my_set #{(1, 2, 3, 4)}
# you can also check if a set is a subset or superset of another set
a_set = {1, 2, 3, 4, 5}
# a subset means it is containted in
{1, 2, 3}.issubset(a_set) # True
# a superset means contains all elements of
a_set.issuperset({1, 2, 3}) # True
# sets are equal if all of their elements are equal
{1, 2, 3} == {3, 2, 1} #True
# List comprehensions allow you to form a new list by filtering and then transforming the elements of a list
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
# the basic form of a list comprehension is [expr for val in collection if condition]
[x.upper() for x in strings if len(x) > 2] # ['BAT', 'CAR', 'DOVE', 'PYTHON']
# the basic form of a set comprehension is {expr for val in collection if condition}
unique_lengths = {len(x) for x in strings}
unique_lengths # {1, 2, 3, 4, 6}
# the map function produces the same result as the set comprehension
set(map(len, strings)) # {1, 2, 3, 4, 6}
# the enumerate function allows us to iterate trhough the list
for index,val in enumerate(strings):
   print(str(index) + "," + val)

#0,a
#1,as
#2,bat
#3,car
#4,dove
#5,python
# create a dictionary with a dict comprehension that creates a lookup map of the strings to their location in the list
loc_mapping = {val : key for key, val in enumerate(strings)}
loc_mapping # {'a': 0, 'as': 1, 'bat': 2, 'car': 3, 'dove': 4, 'python': 5}
# Nested list comprehensions - working with a list inside of another list
all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
            ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
# use a double for loop to go through the nested list comprehension and collect all names with more than two a's
result = [name for names in all_data for name in names if name.count('a') >= 2]
result
# flatten a list of tuples of integers into a simple list
some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
flattened # [1, 2, 3, 4, 5, 6, 7, 8, 9]
# produce a list of lists using a list comprehension inside a list comprehension
[[x for x in tup] for tup in some_tuples] # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# Functions are defined with the def keyword and returned with the return keyword. 
def my_function(x, y, z=1.5):
    if z > 1:
        return z * (x + y)
    else:
        return z / (x + y)

# in my_function, x and y are positional arguments while z is a keyword argument
my_function(5, 6, z=0.7); my_function(3.14, 7, 3.5); my_function(10, 20); 
#0.06363636363636363
#35.49
#45.0
# keywords can also be used to pass positional arguments
my_function(x=5, y=6, z=7) # 77
my_function(y=6, x=5, z=7) # 77
# Functions can access variables in two different scopes: local or global. By default, a variable's scope is local.
def func():
    list_a = []
    for i in range(5):
        list_a.append(i)

func() # list_a is not defined because the local namespace is destroyed after the function is finished
# if we declare the object outside of the function then we can keep it after the function call has completed
list_a = []
def func():
    for i in range(5):
        list_a.append(i)

func()
list_a # [0, 1, 2, 3, 4]
# we can also keep the assigned value by using the global keyword
list_a = None
def func():
    global list_a
    list_a = []
    for i in range(5):
        list_a.append(i)

func()
list_a # [0, 1, 2, 3, 4]
# Python functions can return multiple values
def f():
    a = 5
    b = 6
    c = 7
    return a, b, c

x, y, z = f()
x; y; z; 
#5
#6
#7
# The States data is unclean, it has whitespace, punctuation symbols and mixed case lettering
states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda',
          'south   carolina##', 'West virginia?']
# import the regular expression standard library
import re
# define a clean_strings function which uses separate functions to perform each operation on a new line
def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value)
        value = value.title()
        result.append(value)
    return result

clean_strings(states) # ['Alabama', 'Georgia', 'Georgia', 'Georgia', 'Florida', 'South   Carolina', 'West Virginia']
# an alternative approach is to make a list of the operations
def remove_punctuation(value):
    return re.sub('[!#?]', '', value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result

clean_strings(states, clean_ops) #['Alabama', 'Georgia', 'Georgia', 'Georgia', 'Florida', 'South   Carolina', 'West Virginia']
# functions are also used as arguments to the map function
for x in map(remove_punctuation, states):
    print(x.strip().title())

#Alabama
#Georgia
#Georgia
#Georgia
#Florida
#South   Carolina
#West Virginia
# Lambda (aka Anonymous) Functions
def x_squared(x):
    return x ** 2

x_squared(9) #81
# the same function only without an explicit __name__ attribute
anon_x_squared = lambda x: x ** 2
anon_x_squared(9) #81

strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
# pass a lambda function to the lists sort method in order to sort by the number of distinct letters in the string
strings.sort(key=lambda x: len(set(list(x))))
strings # ['aaaa', 'foo', 'abab', 'bar', 'card']
#Currying: Partial Argument Application
def add_numbers(x, y):
    return x + y

# Currying means deriving new functions from existing ones
add_five = lambda y: add_numbers(5,y)
add_five(10) #15 
# the built-in functools module simplifies Currying with the partial function
from functools import partial
add_five = partial(add_numbers, 5) 
add_five(10) #15 
# Generators are a concise way to create an iterable object
some_dict = {'a': 1, 'b': 2, 'c': 3}
# iterating over a dict yields the keys
for key in some_dict:
    print(key)

#a
#b
#c
# viewing the iterator in the dictionary
dict_iterator = iter(some_dict)
dict_iterator # <dict_keyiterator object at 0x00000191F3DCD9A0>
# most methods expecting a list-like object can accept an iterator including min, max, sum and list
list(dict_iterator) # ['a', 'b', 'c']
# to create a generator, use the yield keyword instead of return in a function
def squares(n=10):
    print('Generating squares from 1 to {0}:'.format(n ** 2))
    for i in range(1, n + 1):
        yield i ** 2

gen = squares()
gen # <generator object squares at 0x00000191F3E63D60>
list(gen) # Generating squares from 1 to 100: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
# initialize the iterator if you have to use it again
gen = squares()
for x in gen:
    print(x, end=' ') #Generating squares from 1 to 100: 1 4 9 16 25 36 49 64 81 100 

# a generator expression is the generator analogue to list, dict and set comprehensions
gen = (x ** 2 for x in range(10))
max(gen) # 81
# a generator expression can be used to make a dictionary
dict((i, i **2) for i in range(5)) #{0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
# use the yeild keyword instead of return to make a generator
def _make_gen():
    for x in range(21):
        yield x ** 2

gen = _make_gen()
y=list(gen) 
y # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400]
max(y) # 400
min(y) # 0
sum(y) # 2870
# the standard library itertools module has a collection of generators for things like groupby, which takes a sequence and a function to group the elements
import itertools
first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names)) # names is a generator

#A ['Alan', 'Adam']
#W ['Wes', 'Will']
#A ['Albert']
#S ['Steven']
# Errors and Exception Handling
float('1.2345') # 1.2345
# float('something') # throws an error ValueError: could not convert string to float: 'something'
# float((1,2)) #Traceback (most recent call last):  File "<stdin>", line 1, in <module> TypeError: float() argument must be a string or a number, not 'tuple'
# in order to avoid the error message we could define our own attempt_float function that doesn't error out on character data
def attempt_float(x):
    try:
        return float(x)
    except:
        return x

attempt_float('1.2345') # 1.2345
attempt_float('something') # something

# catch multiple exception types by writing a tuple of exception types
def attempt_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x

attempt_float((1, 2)) # (1,2)

# Files and the Operating System
path = 'examples/segismundo.txt'
f = open(path)
lines = [x.rstrip() for x in open(path)]
lines[0:2] #['SueÃ±a el rico en su riqueza,', 'que mÃ¡s cuidados le ofrece;']
f.close()
# the with statment automatically closes after the with block so there is no need use the close method
with open(path) as f:
    lines = [x.rstrip() for x in f]

lines[0:2] #['SueÃ±a el rico en su riqueza,', 'que mÃ¡s cuidados le ofrece;']

# reads teh data in UTF-8
f = open(path)
f.read(10) # 'SueÃ±a el '
f2 = open(path, 'rb')  # Binary mode
f2.read(10) #b'Sue\xc3\xb1a el '
# the tell method gives you the current position of the read handle's position
f.tell() #10 
f2.tell() #10
# import the sys module to check the default encoding
import sys
sys.getdefaultencoding() #'utf-8'
# seek changes the file position to the idicated byte
f.seek(3) #3
f.read(1) #'Ã'
f.close()
f2.close()
# write all non-blank lines to a tmp.txt file in the working directory
with open('tmp.txt', 'w') as handle:
    handle.writelines(x for x in open(path) if len(x) > 1)

# open the newly created file and read the lines
with open('tmp.txt') as f:
    lines = f.readlines()

lines[0:2] #['SueÃ±a el rico en su riqueza,', 'que mÃ¡s cuidados le ofrece;']
# import the os module so that we can use the remove method to delete files
import os
os.remove('tmp.txt')
# Bytes and Unicode with Files
with open(path) as f:
    chars = f.read(10)

chars #'SueÃ±a el '
# read the text file in as a binary file
with open(path, 'rb') as f:
    data = f.read(10)

data # b'Sue\xc3\xb1a el '

data.decode('utf8') # 'Sueña el '
# data[:4].decode('utf8') # Traceback (most recent call last):  File "<stdin>", line 1, in <module> UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 3: unexpected end of data
# text mode combined with the encoding option of the open method allows us to write to various encoded files
sink_path = 'sink.txt'
with open(path) as source:
    with open(sink_path, 'xt', encoding='iso-8859-1') as sink:
        sink.write(source.read())

# read the newly created sink.txt file encoded with iso-8859-1
with open(sink_path, encoding='iso-8859-1') as f:
    print(f.read(10))

#delete the sink.txt file
os.remove(sink_path)


# create a 2 x 3 array
import numpy as np
# Generate some random data
data = np.random.randn(2, 3)
data
#array([[ 0.52248239,  1.44647276,  0.21856349],
#       [ 1.84049574, -0.53824593,  0.74871117]])
# multiply all elements by 10
data * 10
#array([[ 5.22482389, 14.46472755,  2.18563487],
#       [18.40495738, -5.38245935,  7.4871117 ]])
#add all elements together
data + data
#array([[ 1.04496478,  2.89294551,  0.43712697],
#       [ 3.68099148, -1.07649187,  1.49742234]])
# create a one dimensional array with the Numerical Python (aka NumPy) package
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1 # array([6. , 7.5, 8. , 0. , 1. ])
# you can create a multidimensional array from nested sequences like a list of lists
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2
#array([[1, 2, 3, 4],
#       [5, 6, 7, 8]])
# the ndim, shape and dtype attributes return the number of rows, rows x columns and the data type, respectively
arr2.ndim # 2 (rows)
arr2.shape #(2,4) (rows,columns)
# possible numpy datatypes are int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, float128, complex64, complex128, complex256, bool, object, string_ or unicode_
arr1.dtype # dtype('float64')
arr2.dtype # dtype('int32')
# there are several array creation functions used to make specific arrays, such as the identity with 1s on the diagonal, or to copy the contents or the shape of another array
np.zeros(10) # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
np.zeros((3, 6))
# array([[0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0.]])
np.empty((2, 3, 2))
np.ones_like(arr1) # array([1., 1., 1., 1., 1.])
np.identity(3)
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])
np.arange(15) # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
### Data Types for ndarrays
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
arr1.dtype #dtype('float64')
arr2.dtype #dtype('int32')
# cast an array from one dtype to another using ndarrays astype method
arr = np.array([1, 2, 3, 4, 5])
arr.dtype #dtype('int32')
float_arr = arr.astype(np.float64)
float_arr.dtype #dtype('float64')
# be aware that if you cast float to integer the decimal part will be left out
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr #array([ 3.7, -1.2, -2.6,  0.5, 12.9, 10.1])
arr.astype(np.int32) #array([ 3, -1, -2,  0, 12, 10])
# converting strings numbers to numeric numbers with the .astype(float) method
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float) #array([ 1.25, -9.6 , 42.  ])
# you can also cast by specifying the array you want the data type to be like instead of specifying  a specific dtype
int_array = np.arange(10)
int_array.dtype #dtype('int32')
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype) #array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
# calling astype will create a new array, hence the need for the double .dtype method call
int_array.astype(calibers.dtype).dtype #dtype('float64')
# note that there are also shorthand dtypes, such as the type code u4 which means means uint32 - an unsigned 32-bit integer type
empty_uint32 = np.empty(8, dtype='u4')
empty_uint32 # array([1, 2, 3, 4, 5, 6, 7, 8], dtype=uint32)
### Arithmetic with NumPy Arrays
# numpy vectorization allows you to express batch operations without having to use any for loops. Arithmetic operations between same sized arrays occurs element-wise.
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
#array([[1., 2., 3.],
#       [4., 5., 6.]])
arr * arr
#array([[ 1.,  4.,  9.],
#       [16., 25., 36.]])
arr - arr
#array([[0., 0., 0.],
#       [0., 0., 0.]])
1 / arr
#array([[1.        , 0.5       , 0.33333333],
#       [0.25      , 0.2       , 0.16666667]])
arr ** 0.5
#array([[1.        , 1.41421356, 1.73205081],
#       [2.        , 2.23606798, 2.44948974]])
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr2
#array([[ 0.,  4.,  1.],
#       [ 7.,  2., 12.]])
arr2 > arr
#array([[False,  True, False],
#       [ True, False,  True]])
### Basic Indexing and Slicing
arr = np.arange(10)
arr #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr[5] #5
arr[5:8] #array([5, 6, 7])
# if you assign a scalar to a slice like this the value is broadcasted to the entire selection of the original array
arr[5:8] = 12
arr #array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])
# taking the slice again shows that the modification is reflected in the source array
arr_slice = arr[5:8]
arr_slice #array([12, 12, 12])
# if we modify a slice of the slice then the mutations are reflected in the origional source array arr
arr_slice[1] = 12345
arr #array([0,1,2,3,4,12,12345,12,8,9])
# the bare slice operator [:] assigns values the same value throughout the entire array slice
arr_slice[:] = 64
arr # array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])
# if you want a copy and not a view use the copy() method. Note that the 128 doesn't get assigned back to the source array when we do it this way.
arr_slice2=arr[5:8].copy()
arr_slice2 #array([64, 64, 64])
arr_slice2[:]=128
arr #array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])
# two dimensional arrays are such that the elements at each index are one-dimensional arrays (lists in this example)
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2] # array([7, 8, 9])
# accessing individual elements recursively
arr2d[0][2] #3
# an alternative method of accessing elements is with a comma-separated list of indices
arr2d[0, 2] #3

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
#array([[[ 1,  2,  3],
#        [ 4,  5,  6]],
#       [[ 7,  8,  9],
#        [10, 11, 12]]])
arr3d[0]
#array([[1, 2, 3],
#       [4, 5, 6]])
# you can assign both scalar values or arras to an element of a 3d array. Assigning the scalar 42 spreads this number to all elements of the list.
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
#array([[[42, 42, 42],
#        [42, 42, 42]],
#       [[ 7,  8,  9],
#        [10, 11, 12]]])
# we can reassign the old values back to the array so that it remains unchanged
arr3d[0] = old_values
arr3d
#array([[[ 1,  2,  3],
#        [ 4,  5,  6]],
#       [[ 7,  8,  9],
#        [10, 11, 12]]])
# using a comma separated list to slice, this says to take the first element (position 0) of the second row (position 1)
arr3d[1, 0] # array([7, 8, 9])
# we can achieve the same results, taking the first element of the second row, by indexing in two steps
x = arr3d[1]
x
#array([[ 7,  8,  9],
#       [10, 11, 12]])
x[0] #array([7, 8, 9])
          
#### Indexing with slices
# one-dimensional ndarrays can be sliced with the same syntax as with lists
arr #array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])
# take position 2 to 7 (not including 7)
arr[1:6] #array([ 1,  2,  3,  4, 64])
# slicing two-dimensional arrays is different
arr2d
#array([[1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 9]])
# select the frist two rows of arr2d
arr2d[:2]
#array([[1, 2, 3],
#       [4, 5, 6]])
# select the first two rows, and from that we do not take position 0 (take the second column onwards) 
arr2d[:2, 1:]
#array([[2, 3],
#       [5, 6]])
# take the second element (position 1) and then from there take everything up to the third element (position 0 and 1)
arr2d[1, :2] #array([4, 5])
# first we take up to the second element (position 0 and 1), and from there take the thrid element (position 2)
arr2d[:2, 2]
#array([3, 6])
# take all rows, but in the nested list only take the first element (position 0)
arr2d[:, :1]
#array([[1],
#       [4],
#       [7]])
# assigning to a slice experssion will assign the value to the whole selection
arr2d[:2, 1:] = 0
arr2d
#array([[1, 0, 0],
#       [4, 0, 0],
#       [7, 8, 9]])


### Boolean Indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names #array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')
data
#array([[ 0.29353275, -0.46234592, -0.98059036, -0.07976851],
#       [-0.5396112 ,  1.53302273, -0.20015644,  0.19274499],
#       [ 1.25002541,  0.35926976, -0.50318698, -0.43216709],
#       [-1.0807837 ,  0.43623544,  1.22679243, -0.04425051],
#       [ 1.05313594, -1.45063462,  2.24046604, -1.0158813 ],
#       [ 0.59746729, -0.58091359, -0.07061595,  0.49722188],
#       [ 0.39073427,  0.48976425,  1.56803019,  1.18032821]])
# comparing names with the string Bob yields a boolean array
names == 'Bob' #array([ True, False, False,  True, False, False, False])
# the boolean array works because the names array has seven elements and the data array has seven rows
data[names == 'Bob']
#array([[ 0.29353275, -0.46234592, -0.98059036, -0.07976851],
#       [-1.0807837 ,  0.43623544,  1.22679243, -0.04425051]])
# comparisions with arrays are vectoized, and we ca; also add slicing in the same step
data[names == 'Bob', 2:]
#array([[-0.98059036, -0.07976851],
#       [ 1.22679243, -0.04425051]])
data[names == 'Bob', 3]
#array([-0.07976851, -0.04425051])
# selecting everything but Bob with teh not equals operator
names != 'Bob' #array([False,  True,  True, False,  True,  True,  True])
# another way to invert a condition is to use the ~ operator
data[~(names == 'Bob')]
#array([[-0.5396112 ,  1.53302273, -0.20015644,  0.19274499],
#       [ 1.25002541,  0.35926976, -0.50318698, -0.43216709],
#       [ 1.05313594, -1.45063462,  2.24046604, -1.0158813 ],
#      [ 0.59746729, -0.58091359, -0.07061595,  0.49722188],
#      [ 0.39073427,  0.48976425,  1.56803019,  1.18032821]])
# the ~ opartor can be used to invert general conditions. Note the = for assign and == for the arithmetic equivalence comparison
cond = names == 'Bob'
data[~cond]
#array([[-0.5396112 ,  1.53302273, -0.20015644,  0.19274499],
#       [ 1.25002541,  0.35926976, -0.50318698, -0.43216709],
#       [ 1.05313594, -1.45063462,  2.24046604, -1.0158813 ],
#       [ 0.59746729, -0.58091359, -0.07061595,  0.49722188],
#       [ 0.39073427,  0.48976425,  1.56803019,  1.18032821]])
# conditions ccan be combined with boolean arithmetic operators & (and) and | (or)
mask = (names == 'Bob') | (names == 'Will')
mask # #array([ True, False,  True,  True,  True, False, False])
data[mask]
#array([[ 0.29353275, -0.46234592, -0.98059036, -0.07976851],
#       [ 1.25002541,  0.35926976, -0.50318698, -0.43216709],
#       [-1.0807837 ,  0.43623544,  1.22679243, -0.04425051],
#       [ 1.05313594, -1.45063462,  2.24046604, -1.0158813 ]])
# boolean arrays can be used to set values, such as setting all negative values to zero (negative dollar amounts are often times removed from modelling data)
data[data < 0] = 0
data
#array([[0.29353275, 0.        , 0.        , 0.        ],
#       [0.        , 1.53302273, 0.        , 0.19274499],
#       [1.25002541, 0.35926976, 0.        , 0.        ],
#       [0.        , 0.43623544, 1.22679243, 0.        ],
#       [1.05313594, 0.        , 2.24046604, 0.        ],
#       [0.59746729, 0.        , 0.        , 0.49722188],
#       [0.39073427, 0.48976425, 1.56803019, 1.18032821]])
# one-dimensional boolean arrays can be used to set whole rows or columns. In this case, not Joe occurs in the first, third, fourth and fifth element, and so 7 is assigned at these rows.
data[names != 'Joe'] = 7
data
#array([[7.        , 7.        , 7.        , 7.        ],
#       [0.        , 1.53302273, 0.        , 0.19274499],
#       [7.        , 7.        , 7.        , 7.        ],
#       [7.        , 7.        , 7.        , 7.        ],
#       [7.        , 7.        , 7.        , 7.        ],
#       [0.59746729, 0.        , 0.        , 0.49722188],
#       [0.39073427, 0.48976425, 1.56803019, 1.18032821]])

### Fancy Indexing
# fancy indexing refers to indexing using integer arrays
arr = np.empty((8, 4))
for i in range(8):
       arr[i] = i

arr
#array([[0., 0., 0., 0.],
#       [1., 1., 1., 1.],
#       [2., 2., 2., 2.],
#       [3., 3., 3., 3.],
#       [4., 4., 4., 4.],
#       [5., 5., 5., 5.],
#       [6., 6., 6., 6.],
#       [7., 7., 7., 7.]])
# to select out a subset of rows in a specific order, you pass a list or ndarray of integers specifying the order
arr[[4, 3, 0, 6]]
#array([[4., 4., 4., 4.],
#       [3., 3., 3., 3.],
#       [0., 0., 0., 0.],
#       [6., 6., 6., 6.]])
# using negative indices selects rows from the end. Here we start with three from the bottom, then five from the bottom then seven from the bottom
arr[[-3, -5, -7]]
#array([[5., 5., 5., 5.],
#       [3., 3., 3., 3.],
#       [1., 1., 1., 1.]])
# create anohter 8x4 array using the arange array creation function
arr = np.arange(32).reshape((8, 4))
arr
#array([[ 0,  1,  2,  3],
#       [ 4,  5,  6,  7],
#       [ 8,  9, 10, 11],
#       [12, 13, 14, 15],
#       [16, 17, 18, 19],
#       [20, 21, 22, 23],
#       [24, 25, 26, 27],
#       [28, 29, 30, 31]])
# passing multiple index arrays allows us to select a one-dimensional array of elements representing each tuple of indicies 4 is in position (1,0), 23 is in position (5,3), and so on.
arr[[1, 5, 7, 2], [0, 3, 1, 2]] #array([ 4, 23, 29, 10])
# in order to return a multidimensional array you have to use the bare slice and double square brackets. Here we start by taking row 2 (position 1), all columns, but we change the order to first, fourth, second, and then third eleemnt.
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
#array([[ 4,  7,  5,  6],
#       [20, 23, 21, 22],
#       [28, 31, 29, 30],
#       [ 8, 11,  9, 10]])

### Transposing Arrays and Swapping Axes

arr = np.arange(15).reshape((3, 5))
arr
arr.T

arr = np.random.randn(6, 3)
arr
np.dot(arr.T, arr)

arr = np.arange(16).reshape((2, 2, 4))
arr
arr.transpose((1, 0, 2))

arr
arr.swapaxes(1, 2)

## Universal Functions: Fast Element-Wise Array Functions

arr = np.arange(15).reshape((3, 5))
arr
#array([[ 0,  1,  2,  3,  4],
#       [ 5,  6,  7,  8,  9],
#       [10, 11, 12, 13, 14]])
# the T attribute transposes an array
arr.T
#array([[ 0,  5, 10],
#       [ 1,  6, 11],
#       [ 2,  7, 12],
#       [ 3,  8, 13],
#       [ 4,  9, 14]])
arr = np.random.randn(6, 3)
arr
#array([[ 0.82482902,  0.04049784, -1.07605549],
#       [ 0.0056238 , -0.20551922,  0.77800613],
#       [-0.07141027, -0.17985994,  1.56802742],
#       [ 0.84875952, -2.19759838, -0.68917457],
#       [-1.41726407, -0.65048874, -1.1894586 ],
#       [ 0.38828659, -0.21815071, -0.04697943]])
# np.dot is known as the inner matrix dot product
np.dot(arr.T, arr)
#array([[ 3.56527059, -0.98293137,  0.08743226],
#       [-0.98293137,  5.37639183,  1.81300848],
#       [ 0.08743226,  1.81300848,  6.11387933]])
arr = np.arange(16).reshape((2, 2, 4))
arr
#array([[[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7]],
#        [[ 8,  9, 10, 11],
#        [12, 13, 14, 15]]])
# for higher dimensional arrays, transpose will accept a tupe of axis numbers
arr.transpose((1, 0, 2))
#array([[[ 0,  1,  2,  3],
#        [ 8,  9, 10, 11]],
#       [[ 4,  5,  6,  7],
#        [12, 13, 14, 15]]])
arr
#array([[[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7]],
#       [[ 8,  9, 10, 11],
#        [12, 13, 14, 15]]])
#the swapaxes method takes a pair of axis numbers and switches them to rearrange teh data
arr.swapaxes(1, 2)
#array([[[ 0,  4],
#        [ 1,  5],
#        [ 2,  6],
#        [ 3,  7]],
#       [[ 8, 12],
#        [ 9, 13],
#        [10, 14],
#        [11, 15]]])
 
 
## Universal Functions: Fast Element-Wise Array Functions
arr = np.arange(10)
arr
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# square root and the exponential function are simple element-wise transformations
np.sqrt(arr)
#array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
#       2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])
# unary functions, or ufuncs, like add, maximum, sqrt and exp, accept an optional out argument
np.exp(arr)
#array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
#       5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
#       2.98095799e+03, 8.10308393e+03])
x = np.random.randn(8)
y = np.random.randn(8)
x
#array([ 0.50742572, -0.51284844, -1.00526395, -2.49860753, -0.53808498,
#       -1.72860626,  0.0114531 , -0.51914164])
y
#array([ 3.12933496, -0.36090838,  0.04703325, -1.19334313,  1.15626668,
#        0.14802248, -1.01369619,  0.06406296])
# numpy's maximum function takes two arrays as input and computes the element-wise maximium of the elements (like R's pmax function)
np.maximum(x, y)
#array([ 3.12933496, -0.36090838,  0.04703325, -1.19334313,  1.15626668,
#        0.14802248,  0.0114531 ,  0.06406296])
arr = np.random.randn(7) * 5
arr
#array([-5.69881437,  3.15535306, -4.30798758,  5.90180301, -1.4648142 ,
#       -2.62031056, -5.55884052])
# the modf function returns two arrays - a fractional and an the integral part of a floating point array
remainder, whole_part = np.modf(arr)
remainder
#array([-0.69881437,  0.15535306, -0.30798758,  0.90180301, -0.4648142 ,
#      -0.62031056, -0.55884052])
whole_part
#array([-5.,  3., -4.,  5., -1., -2., -5.])
arr
#array([-5.69881437,  3.15535306, -4.30798758,  5.90180301, -1.4648142 ,
#       -2.62031056, -5.55884052])
 
## Array-Oriented Programming with Arrays
points = np.arange(-5, 5, 0.01) # 1000 equally spaced points
# create a grid of values for plotting using the meshgrid function which takes two 1d arrays and produces two 2D matrices representingg all the pairs of (x,y)
xs, ys = np.meshgrid(points, points)
ys
#array([[-5.  , -5.  , -5.  , ..., -5.  , -5.  , -5.  ],
#       [-4.99, -4.99, -4.99, ..., -4.99, -4.99, -4.99],
#       [-4.98, -4.98, -4.98, ..., -4.98, -4.98, -4.98],
#       ...,
#       [ 4.97,  4.97,  4.97, ...,  4.97,  4.97,  4.97],
#       [ 4.98,  4.98,  4.98, ...,  4.98,  4.98,  4.98],
#       [ 4.99,  4.99,  4.99, ...,  4.99,  4.99,  4.99]])
# suppose we want to evaluate sqrt(x ** 2 + y ** 2) accross a grid of vlaues
z = np.sqrt(xs ** 2 + ys ** 2)
z
#array([[7.07106781, 7.06400028, 7.05693985, ..., 7.04988652, 7.05693985,
#        7.06400028],
#       [7.06400028, 7.05692568, 7.04985815, ..., 7.04279774, 7.04985815,
#        7.05692568],
#       [7.05693985, 7.04985815, 7.04278354, ..., 7.03571603, 7.04278354,
#        7.04985815],
#       ...,
#       [7.04988652, 7.04279774, 7.03571603, ..., 7.0286414 , 7.03571603,
#        7.04279774],
#       [7.05693985, 7.04985815, 7.04278354, ..., 7.03571603, 7.04278354,
#        7.04985815],
#       [7.06400028, 7.05692568, 7.04985815, ..., 7.04279774, 7.04985815,
#        7.05692568]])
#import matplotlib.pyplot as plt
#plt.imshow(z, cmap=plt.cm.gray); plt.colorbar() 
#<matplotlib.image.AxesImage object at 0x00000166ED41BAC0>
#<matplotlib.colorbar.Colorbar object at 0x00000166ED47B550>
#plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values") 
#plt.draw()
#plt.close('all')
                                                                                                                   
### Expressing Conditional Logic as Array Operations

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
# suppose we want to take values from xarr only when cond is True, a list comprehension does the trick
result = [(x if c else y)
        for x, y, c in zip(xarr, yarr, cond)]

result #[1.1, 2.2, 1.3, 1.4, 2.5]
# numpy's where function allows us to write this very concisely
result = np.where(cond, xarr, yarr)
result #array([1.1, 2.2, 1.3, 1.4, 2.5])

arr = np.random.randn(4, 4)
arr
#array([[ 0.00859354, -1.09657651,  1.14750448, -0.99270475],
#       [ 0.14306963, -0.590324  ,  0.97639398, -0.08868938],
#       [ 1.25712477, -0.69637457, -0.16818035, -1.45435512],
#       [-1.91623718, -0.24010107, -1.67876727,  2.52690646]])
arr > 0
#array([[ True, False,  True, False],
#       [ True, False,  True, False],
#       [ True, False, False, False],
#       [False, False, False,  True]])
# replace all positive values with 2 and all negative values with -2
np.where(arr > 0, 2, -2)
#array([[ 2, -2,  2, -2],
#       [ 2, -2,  2, -2],
#      [ 2, -2, -2, -2],
#       [-2, -2, -2,  2]])
# the second argument is like an else condition and if you put the original object in there it leaves it unchanged when the condition is false
np.where(arr > 0, 2, arr) # set only positive values to 2
#array([[ 2.        , -1.09657651,  2.        , -0.99270475],
#       [ 2.        , -0.590324  ,  2.        , -0.08868938],
#      [ 2.        , -0.69637457, -0.16818035, -1.45435512],
#       [-1.91623718, -0.24010107, -1.67876727,  2.        ]])
    
### Mathematical and Statistical Methods
arr = np.random.randn(5, 4)
arr
#array([[-1.41127063, -0.03029357,  0.43067475,  0.36532058],
#       [-1.03108375,  0.15141741, -0.78136924,  0.03876025],
#       [ 1.19910151, -1.02366398,  0.27687438,  0.42023381],
#       [-1.56317633, -0.515787  , -0.16210095, -0.47815209],
#       [-0.39417392,  0.23990718,  0.3449246 ,  0.87408154]])
arr.mean() #-0.15248877158245885
np.mean(arr) #-0.15248877158245885
arr.sum() #-3.049775431649177
# functions like mean, std and sum accept an optional axis argument that computes the statistic over the requested axis, resulting in an array of results

arr.mean(axis=1) # compute the mean accross the columns
#array([-0.16139222, -0.40556883,  0.21813643, -0.67980409,  0.26618485])
arr.sum(axis=0) # compute the sum down the rows
#array([-3.20060311, -1.17841996,  0.10900355,  1.22024409])
# cumsum produces the intermediate results before showing the total cumulative sum at the end of hte list
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr.cumsum() #array([ 0,  1,  3,  6, 10, 15, 21, 28], dtype=int32)
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr
#array([[0, 1, 2],
#       [3, 4, 5],
#       [6, 7, 8]])
# perform a cumulative sum down the rows
arr.cumsum(axis=0)
#array([[ 0,  1,  2],
#       [ 3,  5,  7],
#       [ 9, 12, 15]], dtype=int32)
# perform a cumulative product accross the columns
arr.cumprod(axis=1)
#array([[  0,   0,   0],
#      [  3,  12,  60],
#      [  6,  42, 336]], dtype=int32)
 

### Methods for Boolean Arrays
arr = np.random.randn(100)
#sum is often used as a way to count the number of Trues in a boolean array
(arr > 0).sum() # Number of positive values (53 in the first sample of 100 random normal values)
# the any method tests whenther one or more values is True
bools = np.array([False, False, True, False])
bools.any()
True
# the all method checks if every value is True
 bools.all()
False
### Sorting
arr = np.random.randn(6)
arr
#array([ 1.9989683 ,  0.3476939 , -0.13847995,  0.12615114, -1.07125211,
#        0.14856577])
# the sort method sorts the array in-place
arr.sort()
arr
#array([-1.07125211, -0.13847995,  0.12615114,  0.14856577,  0.3476939 ,
#        1.9989683 ])
arr = np.random.randn(5, 3)
arr
#array([[ 1.68804106,  0.44347918, -0.10486119],
#       [ 0.69635703, -1.45703416, -0.26444576],
#       [-0.19615279, -0.437229  , -0.3630979 ],
#      [-0.89231718,  0.06731452, -0.50317298],
#       [-2.99312525, -1.33157706, -0.55656801]])
# sort across the columns by passing the axis number to the sort method
arr.sort(1)
arr
#array([[-0.10486119,  0.44347918,  1.68804106],
#       [-1.45703416, -0.26444576,  0.69635703],
#       [-0.437229  , -0.3630979 , -0.19615279],
#       [-0.89231718, -0.50317298,  0.06731452],
#       [-2.99312525, -1.33157706, -0.55656801]])
# a quick way to calculate the quantiles is to sort the array and then select the value at a particular rank
large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))] # 5% quantile (-1.6589863321488525 using the random sample of 1000 normal values)
                  
### Unique and Other Set Logic
# array set operations include unique(x), intersect1d(x,y), union1d(x,y), in1d(x,y), setdiff1d(x,y), setxor1d(x,y)
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names) #array(['Bob', 'Joe', 'Will'], dtype='<U4')
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints) #array([1, 2, 3, 4])
# the set function also forces it's elements to be unique
sorted(set(names)) #['Bob', 'Joe', 'Will']
# the in1d method tests whether the values in one array exist in the other
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6]) #array([ True, False, False,  True,  True, False,  True])
          
## Linear Algebra

x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x
#array([[1., 2., 3.],
#       [4., 5., 6.]])
y
#array([[ 6., 23.],
#       [-1.,  7.],
#       [ 8.,  9.]])
# 1 x 6 + 2 x -1 + 3 x 8 = 28
# the dot for matrix multiplication
x.dot(y)
#array([[ 28.,  64.],
#       [ 67., 181.]])
# the dot array method
np.dot(x, y)
#array([[ 28.,  64.],
#       [ 67., 181.]])
# the @ symbol performs matrix multiplication
x @ np.ones(3)
#array([ 6., 15.])
# import the matrix inverse function and the QR decomposition function from the linalg module
from numpy.linalg import inv, qr
X = np.random.randn(5, 5)
mat = X.T.dot(X)
inv(mat)
#array([[ 1.9353783 , -0.49472235, -0.51236337,  2.05790294, -2.63555533],
#       [-0.49472235,  0.46840973,  0.04402124, -0.72054882,  0.96026777],
#       [-0.51236337,  0.04402124,  0.32358634, -0.56020743,  0.5377636 ],
#       [ 2.05790294, -0.72054882, -0.56020743,  2.91161382, -3.37873216],
#       [-2.63555533,  0.96026777,  0.5377636 , -3.37873216,  4.74388758]])
mat.dot(inv(mat))
#array([[ 1.00000000e+00, -2.09617393e-17,  4.29050416e-17,
#         3.83678817e-16,  7.99175033e-16],
#       [-4.81969935e-16,  1.00000000e+00,  4.67587048e-17,
#         1.18612370e-17,  3.00509589e-16],
#       [ 7.06653787e-16, -4.22863519e-16,  1.00000000e+00,
#         1.29201326e-16,  7.72713054e-17],
#       [-1.01942303e-15,  1.01420896e-16, -2.91973620e-17,
#         1.00000000e+00,  5.88232444e-16],
#       [-6.47921204e-17, -1.80097135e-18, -5.79227643e-17,
#         3.99795079e-16,  1.00000000e+00]])
q, r = qr(mat)
r
#array([[-4.49737301, -0.94125756, -6.68834982, -1.09771538, -2.39558339],
#       [ 0.        , -4.47094059, -3.25615637, -1.66667102,  0.11798257],
#       [ 0.        ,  0.        , -4.1661302 , -2.24191308, -1.12493059],
#       [ 0.        ,  0.        ,  0.        , -2.27389785, -1.7445739 ],
#       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.15416071]])
                                                                                                                 


## Pseudorandom Number Generation
# obtain a 4x4 array of samples from the standard normal distribution
samples = np.random.normal(size=(4, 4))
samples

# The numpy.random functions include seed, permutation, shuffle, rand, randint, randn, binomial, normal, beta, chisquare, gamma and uniform
import timeit #timeit.py is in the same directory
from random import normalvariate
N = 1000000
# numpy's random module is orders of magnitude faster than Python's built-in random module
import time
start_time=time.perf_counter()
samples = [normalvariate(0, 1) for _ in range(N)]
end_time=time.perf_counter()
timer=end_time - start_time
round(timer,2)

start_time=time.perf_counter()
np.random.normal(size=N)
end_time=time.perf_counter()
timer=end_time - start_time
round(timer,2)

# you can set NumPy's random number generation seed using the random.seed method
np.random.seed(1234)
# data generation functions in numpy.random use a global random seed. The RandomState avoids having a global seed value.
rng = np.random.RandomState(1234)
rng.randn(10)
#array([ 0.47143516, -1.19097569,  1.43270697, -0.3126519 , -0.72058873,
#        0.88716294,  0.85958841, -0.6365235 ,  0.01569637, -2.24268495])
## Example: Random Walks
# the simple random walk starts at zero and goes up one or down one with equal probability at each step
import random
position = 0
walk = [position]
steps = 1000
# the randint function draws random integers from a given low to high range. Here it takes on values 0 or 1 with 50% chance each.
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

# plot the random walk
plt.figure()
plt.plot(walk[:100])

# an equivalent techique to generate a random walk is to draw flips of a coin and use the cumsum function to tally up the location at each step
np.random.seed(12345)
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
# we cam find the farthest away the process walked with the min and max functions
walk.min()
walk.max()
# argmax returns the first index of the maximum value in a boolean array
(np.abs(walk) >= 10).argmax()

### Simulating Many Random Walks at Once
# simulate 5000 random walks of 1000 steps each
nwalks = 5000
nsteps = 1000
# when passed a two-dimensional array, numpy.random functions generate a two-dimensional array of draws
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
# view the objects created
draws
#array([[0, 1, 1, ..., 1, 0, 0],
#       [0, 1, 1, ..., 1, 1, 0],
#       [1, 1, 1, ..., 0, 0, 0],
#       ...,
#       [1, 0, 1, ..., 0, 1, 1],
#       [0, 1, 0, ..., 0, 1, 0],
#       [1, 1, 1, ..., 0, 0, 1]])
steps
#array([[-1,  1,  1, ...,  1, -1, -1],
#       [-1,  1,  1, ...,  1,  1, -1],
#       [ 1,  1,  1, ..., -1, -1, -1],
#       ...,
#       [ 1, -1,  1, ..., -1,  1,  1],
#       [-1,  1, -1, ..., -1,  1, -1],
#       [ 1,  1,  1, ..., -1, -1,  1]])
walks
#array([[ -1,   0,   1, ...,  14,  13,  12],
#       [ -1,   0,   1, ...,  44,  45,  44],
#       [  1,   2,   3, ..., -18, -19, -20],
#       ...,
#       [  1,   0,   1, ...,   4,   5,   6],
#       [ -1,   0,  -1, ..., -12, -11, -12],
#       [  1,   2,   3, ...,  46,  45,  46]], dtype=int32)

walks.max() #116
walks.min() #-116
# out of these walks, let's find the minimum crossing time to 30 or -30
hits30 = (np.abs(walks) >= 30).any(1)
# not all random walks reached beyond +30 or -30
hits30 #array([False,  True,  True, ..., False, False,  True])                                                   
hits30.sum() # Number that hit 30 or -30 (3401 in the first test)
# using the boolean array we can select out the walks taht reached +/-30 and call the argmax across axis 1 to get the crossing times
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times #array([311, 673, 349, ..., 591, 937, 763], dtype=int64)
crossing_times.mean() #498.74
# other probability distributions could be used for the random walk. Another possibility is to use a normal distribution with a mean of 0 and a stdev of .25
steps = np.random.normal(loc=0, scale=0.25,
                         size=(nwalks, nsteps))
steps
#array([[-0.05015418, -0.25338291,  0.4310268 , ...,  0.08039154,
#        -0.3318606 ,  0.14001608],
#       [ 0.07460316,  0.01934373, -0.03534487, ..., -0.00131745,
#        -0.21087354, -0.07395815],
#       [-0.08484035,  0.06263097,  0.36061029, ...,  0.11130218,
#        -0.08953152,  0.32025783],
#       ...,
#       [ 0.55506066,  0.21128525, -0.25303571, ..., -0.21399762,
#         0.25019564,  0.30824294],
#       [ 0.06752211,  0.39087401,  0.10655129, ..., -0.22253245,
#         0.05066762, -0.20347334],
#       [ 0.03302648,  0.14489699,  0.05726634, ..., -0.09882506,
#         0.02203751,  0.02237818]])


## Introduction to pandas Data Structures
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
### Series
# a series is a one-dimensional array like object containing a sequence of values
obj = pd.Series([4, 7, -5, 3])
obj
#0    4
#1    7
#2   -5
#3    3
#dtype: int64

obj.values
array([ 4,  7, -5,  3], dtype=int64)
obj.index  # like range(4)
RangeIndex(start=0, stop=4, step=1)

# the index can be character labels (whereas in NumPy arrays the index must be integer based)
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
#d    4
#b    7
#a   -5
#c    3
#dtype: int64

obj2.index
#Index(['d', 'b', 'a', 'c'], dtype='object')

obj2['a']
#-5
# you can change a NumPy series in place
obj2['d'] = 6
obj2[['c', 'a', 'd']] # viewing the values for several indices
#c    3
#a   -5
#d    6
#dtype: int64

# NumPy-like operations can be used
obj2[obj2 > 0]
#d    6
#b    7
#c    3
#dtype: int64
obj2 * 2
#d    12
#b    14
#a   -10
#c     6
#dtype: int64
np.exp(obj2)
#d     403.428793
#b    1096.633158
#a       0.006738
#c      20.085537
#dtype: float64

'b' in obj2
#True
'e' in obj2
#False

# a series can be thought of as a dictionary
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3
#Ohio      35000
#Texas     71000
#Oregon    16000
#Utah       5000
#dtype: int64

# you can override the keys being in sorted order by passing in the dictionary keys in the order that you want
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4
#California        NaN
#Ohio          35000.0
#Oregon        16000.0
#Texas         71000.0
#dtype: float64
                                                                                                                    
# missing or NA appear as NaN (Not a Number) in pandas and the isnull and notnull functions in pandas can be used to detect missing data
pd.isnull(obj4)
#California     True
#Ohio          False
#Oregon        False
#Texas         False
#dtype: bool
#pd.notnull(obj4)
#California    False
#Ohio           True
#Oregon         True
#Texas          True
#dtype: bool

# the Series stores this information as instance methods
obj4.isnull()
#California     True
#Ohio          False
#Oregon        False
#Texas         False
#dtype: bool

obj3
#Ohio      35000
#Texas     71000
#Oregon    16000
#Utah       5000
#dtype: int64
obj4
#California        NaN
#Ohio          35000.0
#Oregon        16000.0
#Texas         71000.0
#dtype: float64
obj3 + obj4
#California         NaN
#Ohio           70000.0
#Oregon         32000.0
#Texas         142000.0
#Utah               NaN
#dtype: float64

# both the series object itself and its index have a name attribute
obj4.name = 'population'
obj4.index.name = 'state'
obj4
#state
#California        NaN
#Ohio          35000.0
#Oregon        16000.0
#Texas         71000.0
#Name: population, dtype: float64

# a Series's index can be altered in-place
obj
#0    4
#1    7
#2   -5
#3    3
#dtype: int64

obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj
#Bob      4
#Steve    7
#Jeff    -5
#Ryan     3
#dtype: int64
                                                                                                                    

### DataFrame

# one of the most popular ways to create a DataFrame is from a dictionary of equal-length lists
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
         'year': [2000, 2001, 2002, 2001, 2002, 2003],
         'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
# the panda dataframe object has it's index automatically assigned
frame = pd.DataFrame(data)
#the head method selects only the first five rows
frame.head()
    #state  year  pop
#0    Ohio  2000  1.5
#1    Ohio  2001  1.7
#2    Ohio  2002  3.6
#3  Nevada  2001  2.4
#4  Nevada  2002  2.9

# you can rearrange the column ordering by passing a sequence of columns
pd.DataFrame(data, columns=['year', 'state', 'pop'])
   #year   state  pop
#0  2000    Ohio  1.5
#1  2001    Ohio  1.7
#2  2002    Ohio  3.6
#3  2001  Nevada  2.4
#4  2002  Nevada  2.9
#5  2003  Nevada  3.2

# passing in a missing column causes it to appear as missing in the resulting dataframe
# we are also creatinga  new index here
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                       index=['one', 'two', 'three', 'four','five', 'six'])
frame2
#       year   state  pop debt
#one    2000    Ohio  1.5  NaN
#two    2001    Ohio  1.7  NaN
#three  2002    Ohio  3.6  NaN
#four   2001  Nevada  2.4  NaN
#five   2002  Nevada  2.9  NaN
#six    2003  Nevada  3.2  NaN

frame2.columns
#Index(['year', 'state', 'pop', 'debt'], dtype='object')

# a column can be retrieved as a series using the dictionary-like notation
frame2['state']
#one        Ohio
#two        Ohio
#three      Ohio
#four     Nevada
#five     Nevada
#six      Nevada
#Name: state, dtype: object

# retrieving a column as a series by attribute
frame2.year
#one      2000
#two      2001
#three    2002
#four     2001
#five     2002
#six      2003
#Name: year, dtype: int64

# retrieve row 3 by name with the loc attribute
frame2.loc['three']
#year     2002
#state    Ohio
#pop       3.6
#debt      NaN
#Name: three, dtype: object

# Columns can be modified by assignment, either a scalar value or an array of values
frame2['debt'] = 16.5
frame2
       #year   state  pop  debt
#one    2000    Ohio  1.5  16.5
#two    2001    Ohio  1.7  16.5
#three  2002    Ohio  3.6  16.5
#four   2001  Nevada  2.4  16.5
#five   2002  Nevada  2.9  16.5
#six    2003  Nevada  3.2  16.5

frame2['debt'] = np.arange(6.)
frame2
       #year   state  pop  debt
#one    2000    Ohio  1.5   0.0
#two    2001    Ohio  1.7   1.0
#three  2002    Ohio  3.6   2.0
#four   2001  Nevada  2.4   3.0
#five   2002  Nevada  2.9   4.0
#six    2003  Nevada  3.2   5.0

# you can join rows based on their index value and any non-matching values will be left as missing
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
#       year   state  pop  debt
#one    2000    Ohio  1.5   NaN
#two    2001    Ohio  1.7  -1.2
#three  2002    Ohio  3.6   NaN
#four   2001  Nevada  2.4  -1.5
#five   2002  Nevada  2.9  -1.7
#six    2003  Nevada  3.2   NaN

# create a boolean column named eastern that is True when the state is "Ohio" and False when it is not equal to "Ohio"
frame2['eastern'] = frame2.state == 'Ohio'
frame2
#       year   state  pop  debt  eastern
#one    2000    Ohio  1.5   NaN     True
#two    2001    Ohio  1.7  -1.2     True
#three  2002    Ohio  3.6   NaN     True
#four   2001  Nevada  2.4  -1.5    False
#five   2002  Nevada  2.9  -1.7    False
#six    2003  Nevada  3.2   NaN    False

# the del method can be used to drop a column
del frame2['eastern']
frame2.columns
#Index(['year', 'state', 'pop', 'debt'], dtype='object')

# if a nested dict is passed then pandas interprets the outer keys as columns and the inner keys as the row indices
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

frame3 = pd.DataFrame(pop)
frame3
#         Nevada  Ohio
#2001     2.4   1.7
#2002     2.9   3.6
#2000     NaN   1.5
               
# you can transpose a DataFrame with the .T function
frame3.T   
#         2001  2002  2000
#Nevada   2.4   2.9   NaN
#Ohio     1.7   3.6   1.5
You can specify your own index to override the default way that the keys in the the inner dicts are combined and sorted
pd.DataFrame(pop, index=[2001, 2002, 2003])
#state Nevada  Ohio 
#year
#2001     2.4   1.7  
#2002     2.9   3.6   
#2003     NaN   NaN  

#Dicts of Series are similar to nested Dicts
pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
pd.DataFrame(pdata)
#       Ohio  Nevada
#2001   1.7     2.4
#2002   3.6     2.9

# You can display a DataFrame's index and column names
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3  
#state  Nevada  Ohio     
#year 
#2001      2.4   1.7   
#2002      2.9   3.6  
#2000      NaN   1.5    

# the values attribute returns the data as a 2-D array
frame3.values

### Index Objects hold axis labels or names and other metadata and theya re immutable

obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index
#Index(['a', 'b', 'c'], dtype='object')
index[1:]
#Index(['b', 'c'], dtype='object')

# index[1] = 'd'  # TypeError
# Immutability means that it safter to share index objects
labels = pd.Index(np.arange(3))
labels
#0    1.5  
#1   -2.5   
#2    0.0 
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2
obj2.index is labels
#True
# an Index behaves like a Series
frame3
#state  Nevada  Ohio
#year
#2001      2.4   1.7
#2002      2.9   3.6
#2000      NaN   1.5
frame3.columns
#Index(['Nevada', 'Ohio'], dtype='object', name='state')
'Ohio' in frame3.columns
#True
2003 in frame3.index
#False
# unlike python sets, the Pandas Index can have duplicate values
dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
dup_labels
#Index(['foo', 'foo', 'bar', 'bar'], dtype='object')
