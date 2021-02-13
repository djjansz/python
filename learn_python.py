phrase="Dave J"
print(phrase.upper().isupper())
print(len(phrase))
print(phrase[3])
print(phrase.index("J"))
print(phrase.replace("Dave ","D"))
print(10%3)
print(pow(10,3))
print(min(10,3))
print(round(3.2))
print(str(88) + " symbolizes fortune and good luck")

from math import *
print(ceil(3.2))
print(sqrt(36))

name = input("Enter you name: ")
print("Hello " + name)

# the numbers are concatenated
num1 = input("Enter a number: ")
num2 = input("Enter another number: " )
result = num1 + num2
print(result)
# whole numbers are summed
result = int(num1) + int(num2)
print(result)
# decimal numbers are summed
result = float(num1) + float(num2)
print(result)


list_x = ["Dave",37,True]
list_x[0]
list_x[-1]
list_x[1:]
# update element one
list_x[0]="David Jeremy"
list_x


lucky_numbers=[2,8,12,18,1]
friends=["Eugene","Ivan","Denny"]
friends.extend(lucky_numbers)
print(friends)

friends.extend(lucky_numbers)
print(friends)
['Eugene', 'Ivan', 'Denny', 1, 8, 12, 18, 21]
friends.append("Mark")
print(friends)
# insert at position one and bump the rest to the right
friends.insert(1,"Mario")
# pop off the last one 
friends.pop()
# show the position where Mario is in the list
print(friends.index("Mario"))
# count the number of times Denny appears
print(friends.count("Denny"))
# sort the lucky numbers list by ascending order
lucky_numbers.sort()
print(lucky_numbers)
lucky_numbers.reverse()
print(lucky_numbers)
friends2 = friends.copy()
print(friends2)

# tuples are immutable
coord inates = (44,79)
# does not allow it to be changed
coordinates[1] = 10
print(coordinates[1])

# define a function with one parametre
def printHello(name):
          print("Hello " + name)

printHello("Dave")

# take the cube aka the power of three
def cube(num):
       return num*num*num
result=cube(4)	   
print(result)

#find the max of three numbers
def max_num(num1,num2,num3):
        if num1>=num2 and num1>= num3:
            return num1
        elif num2>=num1 and num2>=num3:
             return num2
        else:
             return num3
 
max_num(7,8,2)

# month dictionary - the shorthand is the key, long name is the value
monthLookup = {
    "Jan" : "January",
    "Feb" : "February", 
    "Mar" : "March",
     "May" : "May",
    "Jun" : "June",
    "Jul" : "July",
    "Aug" : "August",
    "Sep" : "September",
    "Oct" : "October",
    "Nov" : "November",
    "Dec" : "December"
}

print(monthLookup["Dec"])
print(monthLookup.get("Dc","Not a valid key."))

#  while loop
i = 1;
while i<=10:
    print(i);
    i += 1;
 
# Generate psuedorandom numbers
from random import seed
from random import random
from math import floor
seed(12345)
secret_num=floor(random()*100)
guess = 0
guess_count=0
guess_limit=5
out_of_guesses = False

while guess != secret_num and not(out_of_guesses):
    if guess_count < guess_limit:
         guess = input("Enter guess: ")
         guess_count += 1
    else:
         out_of_guesses = True

if out_of_guesses: 
    print("Out of guesses, the number was " + str(secret_num))
else:
    print("You win, the number was " + str(secret_num))


friends = ["Denny","Ivan","Eugene"]
for friend in friends:
    print(friend)

#range(0,3) is 0,1,2
for i in range(0,len(friends)):
    print(i)
      
# print the numbers 3 to 10      
for index in range(3,10):
    print(index)
    