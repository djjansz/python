# Generate psuedorandom numbers
from random import seed
from random import random
from math import floor
input_seed=input("Enter seed: ")
secret_num=floor(random()*100)
guess = 0
guess_count=0
guess_limit=5
out_of_guesses = False

while guess != secret_num and not(out_of_guesses):
    if guess_count < guess_limit:
         guess = input("Enter guess (1 to 100): ")
         guess_count += 1
    else:
         out_of_guesses = True

if out_of_guesses: 
    print("Out of guesses, the number was " + str(secret_num))
else:
    print("You win, the number was " + str(secret_num))
