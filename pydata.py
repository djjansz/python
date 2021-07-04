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


