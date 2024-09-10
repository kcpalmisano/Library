# -*- coding: utf-8 -*-
"""
Function practice and training 
"""


def greet(name, greeting="Hola"):
    return f"{greeting}, {name}!"

print(greet("Alice"))
#Hola, Alice!

greet("Jenn", greeting="hello")
#'hello, Jenn!'

print(greet(name="Casey", greeting="Hi"))
##Hi, Casey!

###=====================================
'''
List:
A list is a mutable, ordered collection that can hold elements of different data types. Lists are defined with square brackets [].
my_list = [1, 2, 3, "apple", "banana"]

Array:
In Python, arrays are typically implemented using libraries like NumPy because Python’s built-in lists are used in a similar way. NumPy arrays are more efficient for numerical operations.
my_array = np.array([1, 2, 3, 4])

 Dictionary:
A dictionary is a mutable, unordered collection of key-value pairs. Dictionaries are defined using curly braces {}.
my_dict = {"name": "John", "age": 30, "city": "New York"}
'''

##positional arguments
def print_args(*args):
    for arg in args: 
        print(arg)
        
print_args(1,2,3)
'''
1
2
3
'''
        
##variable number of keyword arguments
def print_kwargs(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_kwargs(a=1, b=2, c=3)
'''
a: 1
b: 2
c: 3
'''

###=====================================

add = lambda x, y: x + y

print(add(2,3))
## 5


numbers = [1,2,3,4]
squared = list(map(lambda x: x**2, numbers))

print(squared) 
##[1, 4, 9, 16]


###!!! =====================================
### Create a function that will take a list of numbers and square or cube them

nums = [1,2,3,4,5]

def power_dict(numbers, power_by = 2):
    return {num: num**power_by for num in numbers}


power_dict(nums, power_by=2)
##{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}


power_dict(nums, power_by=3)
##{1: 1, 2: 8, 3: 27, 4: 64, 5: 125}


###=====================================
### Filter even and odd numbers 

nums = [1,2,3,4,5,6,7,8,9,10]

def filter_numbers (numbers, filter_by = 'even'):
    if  filter_by == 'even':
        return [num for num in numbers if num % 2 == 0 ]
    elif  filter_by == 'odd':
        return [num for num in numbers if num % 2 != 0 ]
    else: 
        raise ValueError("filter_type must be 'even' or 'odd'.") 
    
filter_numbers(nums)
##[2, 4, 6, 8, 10]

odds = filter_numbers(nums, filter_by = 'odd')
##[1, 3, 5, 7, 9]


###=====================================
## Apply a function to each element

def apply_function(numbers, func): 
    return [func(num) for num in numbers]
  

sqrd = apply_function(nums, lambda x:x**2)   
    
###=====================================
## Sum a dictionary 

def sum_values(dic):
    return sum(dic.values())
    
data = {'a': 10, 'b': 20, 'c': 30}   

ttl_sum = sum_values(data)    
    

###=====================================
## dictionary of factorials
import math

def factorial_dict(numbers):
    return{num: math.factorial(num) for num in numbers}

nums = [1,2,3,4,5]

factorial_dict(nums) 
##  {1: 1, 2: 2, 3: 6, 4: 24, 5: 120}

'''
List Comprehensions
[expression for item in iterable]

[num ** 2 for num in numbers]  ##Squares each number in the list

#Explaination
for num in numbers: This part iterates over each item in the numbers list.
num ** 2: This is the expression applied to each item. It squares the current item num.
[ ... ]: The square brackets indicate that a list is being created.

## Use an 'if' to filter 
[num **2 for num in numbers if num % 2 ==0]  ##only squares even numbers 
--------------------------

Dicitonary Comprehensions
{key_expression: value_expresson for item in iterable}

{num: num **2 for num in numbers}  ##creates a dictionary with numbers as keys and their squares as values

#Explaination
num: num ** 2: This creates a key-value pair in the dictionary where the key is num and the value is num ** 2.
for num in numbers: This iterates over each item in the numbers list.
{ ... }: The curly braces indicate that a dictionary is being created.

## Use an 'if' to filter 
{num: num **2 for num in numbers if num %2 ==0} ##only squares even numbers
-----------------
 
Set Comprehensions
{expression for item in iterable}

{num ** 2 for num in numbers} ##Squares numbers and remove dupes 

--------------------------
##Nested Comprehensions 
matrix = [[1,2,3], [4,5,6],[7,8,9]]
flat_list = [num for row in matrix for num in rows]
## Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]
'''

###===================================

##Task: Given a list of numbers, create a new list containing the squares of these numbers using a list comprehension.
sqr = [num **2 for num in nums]
print(sqr) 
##[1, 4, 9, 16, 25]

##Task: Given a list of numbers, create a new list that only contains the even numbers using a list comprehension.
numbers = [1,2,3,4,5,6,7,8,9,10]
evens = [num for num in numbers if num % 2 == 0]
##[2, 4, 6, 8, 10]

##Task: Given a list of numbers, create a dictionary where each number is a key and its square is the value.
diction = {num: num **2 for num in nums}
##{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

##Task: Given a list of numbers, create a new list that contains the squares of even numbers only.
sqr_evens = {num: num**2 for num in numbers if num %2 ==0}
##{2: 4, 4: 16, 6: 36, 8: 64, 10: 100}

##Task: Given a 2D list (a list of lists), create a flattened version of it using a list comprehension.
matrix = [[1,2,3], [4,5,6]]
flat = [num for rows in matrix for num in rows]
##[1, 2, 3, 4, 5, 6]

##Task: Given a dictionary, create a new dictionary where the keys are the values from the original dictionary and the values are the keys.
original_dict = {'a': 1, 'b': 2, 'c': 3}
rev_dict = {value: key for key, value in original_dict.items()}
##{1: 'a', 2: 'b', 3: 'c'}

##Task: Given a list of numbers, create a new list that contains only the numbers that are multiples of 3 or 5.
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
t_and_f = [num for num in numbers if num % 3 == 0 or num % 5 == 0] 
##[3, 5, 6, 9, 10, 12, 15]

##Task: Given a string, create a dictionary that maps each character to the number of times it appears in the string.
text = 'hello world'
count = {char: text.count(char) for char in set(text)}
##{'w': 1, 'h': 1, 'e': 1, 'r': 1, ' ': 1, 'd': 1, 'l': 3, 'o': 2}

##Task: Given a list of lists, create a flattened list that only includes the elements greater than 10.
nested_lists = [[5, 12, 17], [8, 10, 14], [7, 19, 22]]
high = [num for sublist in nested_lists for num in sublist if num > 10]
##[12, 17, 14, 19, 22]

##Task: Given a list of words, create a dictionary where the keys are the words and the values are the lengths of those words.
words = ["apple", "banana", "cherry", "date"]
fruit = {word: len(word) for word in words}
##{'apple': 5, 'banana': 6, 'cherry': 6, 'date': 4}

##Task: Given a string, create a set of all the unique vowels found in the string.
text = "This is a sample string."
vowels = {'a', 'e', 'i', 'o', 'u'}
txt_vowls = {char for char in text.lower() if char in vowels}
##{'i', 'a', 'e'}

#Bonus add in their count
txt_vowls = {char: text.count(char) for char in text.lower() if char in vowels}
##{'i': 3, 'a': 2, 'e': 1}


##Task: Create a list of tuples that pairs each number with its square from a given list.
numbers = [1, 2, 3, 4]
[(num, num**2) for num in numbers]
### [ (element1, element2) for item in iterable ]

##Task: Given a list of numbers, create a dictionary where the keys are 'even' and 'odd', and the values are lists of the corresponding even and odd numbers.
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
{
 'even': [num for num in numbers if num % 2 == 0],
 'odd': [num for num in numbers if num % 2 != 0]
 }

###!!! =====================================
from functools import reduce
##  lambda arguments: expression

##Add ten to the input
add_ten = lambda x: x+10
add_ten(13)
### 23

'''
Common lambda functions: 

map():
map() applies a function to all items in an iterable (e.g., list) and returns a map object (which can be converted into a list).

numbers = [1,2,3,4,5]
squared = list(map(lambda x: x**2, numbers))
## Output [1, 4, 9, 16, 25]


filter():
filter() applies a function to an iterable and returns a new iterable containing only the items for which the function returns True.

numbers = [1, 2, 3, 4, 5]
even_nums = list(filter(lambda x: x % 2 ==0, numbers))
Output [2, 4]

reduce():
reduce() applies a function of two arguments cumulatively to the items of an iterable, reducing the iterable to a single value. reduce() is part of the functools module.1

from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers) 
Output 120        Essentially it does this -> (1*2*3*4*5)

###### Higher-Order Functions
A higher-order function is a function that either:

-Takes one or more functions as arguments.
-Returns a function as its result.

EX:
def apply_operation(func, numbers):
    return [func(num) for num in numbers]

def square(x):
    return x**2

numbers = [1,2,3,4,5]

apply_operation(square, numbers))  
## Output [1, 4, 9, 16, 25]

### Using a lambda function 
apply_operation(lambda x: x+1, numbers)) 
## Output [2, 3, 4, 5, 6]
'''

###Task: Given a list of strings, use map() and a lambda function to convert each string to uppercase.
words = ["hello", "world", "python"]
upper = list(map(lambda x: str.upper(x), words))   #ALSO  list(map(lambda x: x.upper(), words))
##['HELLO', 'WORLD', 'PYTHON']

##Task: Given a list of numbers, use filter() and a lambda function to return only the numbers greater than 5.
numbers = [2, 5, 8, 1, 9, 3, 7]
list(filter(lambda x: x> 5, numbers))
## [8, 9, 7]

##Task: Given a list of numbers, use reduce() and a lambda function to compute the sum of all numbers.
numbers = [1, 2, 3, 4, 5]
reduce(lambda x, y: x+y, numbers)
# 15        #ALSO sum(numbers)

##Task: Write a function make_multiplier that takes a number n as an argument and returns a new function that multiplies its input by n.
def make_multiplier(n):
    return lambda x: x * n

dub = make_multiplier(2)
trip = make_multiplier(3)

dub(10)     #20
trip(13)    #39


###Task: Given a list of numbers, first filter out the numbers that are not divisible by 3, and then square the remaining numbers using map().
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
first = list(filter(lambda x: x%3 ==0, numbers))
second = list(map(lambda x: x**2, first))
## [9, 36, 81]

## BETTER   filtered_and_squared = list(map(lambda x: x ** 2, filter(lambda x: x % 3 == 0, numbers)))


##Task: Given a list of numbers, use reduce() to find the maximum number.
numbers = [2, 5, 8, 1, 9, 3, 7]
reduce(lambda x, y: x if x>y else y, numbers)
## 9

##Task: Given a dictionary where the keys are product names and the values are prices, use map() to apply a discount of 10% to all prices.
prices = {'apple': 1.00, 'banana': 0.50, 'orange': 0.75}
dict(map(lambda item: (item[0], item[1] *.9), prices.items()))
##{'apple': 0.9, 'banana': 0.45, 'orange': 0.675}

##Task: Given a list of words, use filter() to find all the words that are longer than 5 characters, and then use map() to convert these words to uppercase.
##****Filter is done first and is more efficent than to upper all words first
words = ["apple", "banana", "cherry", "date", "fig", "grape"]
list(map(lambda word: word.upper(), filter(lambda word: len(word) > 5, words)))
##['BANANA', 'CHERRY']

##Task: Given a list of lists of numbers, filter out the even numbers from each sublist and then double the remaining numbers using map().
nested_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
[list(map(lambda x: x*2, filter(lambda x: x % 2 != 0, sublist))) for sublist in nested_lists]
#[[2, 6], [10], [14, 18]]

##Task: Write a function add_suffix that takes a string and adds the suffix "_checked". Use map() to apply this function to a list of strings.
strings = ["item1", "item2", "item3"]

def add_suffix(string): 
    print(f"{string}_checked")

list(map(lambda string: add_suffix(string), strings ))
##item1_checked
##item2_checked
##item3_checked

'''
ALSO
def add_suffix(s):
    return s + "_checked"

strings = ["item1", "item2", "item3"]
checked_strings = list(map(add_suffix, strings))
##['item1_checked', 'item2_checked', 'item3_checked']
'''

##Task: Given a list of dictionaries representing products (with name and price), filter out the products that are priced above $20, and then apply a 10% discount to the remaining products' prices.
products = [
    {'name': 'apple', 'price': 10},
    {'name': 'banana', 'price': 25},
    {'name': 'orange', 'price': 15},
    {'name': 'mango', 'price': 30}
]

list(map(lambda product: {'name': product['name'], 'price': product['price'] * .9}, 
         filter(lambda product: product['price'] <= 20, products)))
##[{'name': 'apple', 'price': 9.0}, {'name': 'orange', 'price': 13.5}]


##Task: Given a list of words, use reduce() and a lambda function to concatenate them into a single string, separated by a space.
words = ["Hello", "world", "this", "is", "Python"]

reduce(lambda x, y: x + " " + y, words)
##Out[146]: 'Hello world this is Python'

##Task: Given a list of tuples, where each tuple contains a product name and its price, 
##filter out the products that are not fruits (assume fruits are "apple", "banana", "orange"),
## then increase the price of the remaining products by 20%.
products = [
    ("apple", 10),
    ("carrot", 5),
    ("banana", 8),
    ("broccoli", 7),
    ("orange", 12)
]
fruit = ['apple', 'banana', 'orange']

list(map(lambda product: (product[0], product[1] *1.2), 
         filter(lambda product: product[0] in fruit, products )))
##[('apple', 12.0), ('banana', 9.6), ('orange', 14.399999999999999)]

##Task: Given a list of dictionaries representing students (with name and grade), 
#sort the students first by their grades in descending order, and then by their names alphabetically.
students = [
    {'name': 'Alice', 'grade': 90},
    {'name': 'Bob', 'grade': 85},
    {'name': 'Charlie', 'grade': 90},
    {'name': 'David', 'grade': 75}
]

sorted(students, key = lambda s: (-s['grade'], s['name']))
##[{'name': 'Alice', 'grade': 90}, {'name': 'Charlie', 'grade': 90}, {'name': 'Bob', 'grade': 85}, {'name': 'David', 'grade': 75}]

##** sorted( iterable, key = students list  reverse/desc sort by grade, alphabetical/asc sort by name )

##Task: Given a list of numbers, use map() and a lambda function to return a new list where 
#each number is doubled if it’s even, or tripled if it’s odd.
numbers = [1, 2, 3, 4, 5]

list(map(lambda x: x*2 if x%2 ==0 else x*3, numbers ))
##[3, 4, 9, 8, 15]

