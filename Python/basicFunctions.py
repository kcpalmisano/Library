# -*- coding: utf-8 -*-

### FUNCTIONS 
'''
Syntax elements 

1. Variable Assignment
Assign values to variables using the = operator.
Example: x = 10, name = "Alice"

2. Arithmetic Operations
Perform basic mathematical operations: +, -, *, /, % (modulus), ** (exponentiation).
Example: result = 5 + 3 * 2 (follows order of operations)

3. Comparison Operators
Compare values: ==, !=, <, >, <=, >=
Example: if x > 5:
    
4. Logical Operators
Combine conditions: and, or, not
Example: if x > 5 and y < 10:
    
5. Conditional Statements
Perform actions based on conditions: if, elif, else

Example:    
if x > 10:
    print("x is greater than 10")
elif x == 10:
    print("x is 10")
else:
    print("x is less than 10")
    
6. Loops
Repeat actions: for, while
for loop: Iterate over a sequence (like a list, tuple, dictionary, set, or string).
while loop: Repeat as long as a condition is true.

Example:
for i in range(5):
    print(i)  # Prints 0 to 4

while x < 10:
    x += 1  # Increments x until it reaches 10
    
7. List Comprehensions
Create lists in a concise way.
Example:
    [x ** 2 for x in range(5)] creates [0, 1, 4, 9, 16]

8. Function Definitions
Define reusable code blocks: def function_name(parameters):
    
Example:
def add(a, b):
    return a + b

9. String Operations
Concatenate (+), repeat (*), or format (f"{}") strings.

Example: 
    greeting = "Hello, " + name

10. Slicing
Extract parts of sequences (like strings, lists, tuples): `[start:end:step]`
 Example: `s = "Python"`; `s[1:4]` returns `"yth"`

11. Dictionaries
Key-value pairs accessed using keys.
 Example:
  d = {'a': 1, 'b': 2}
  print(d['a'])  # Outputs 1
  
12. Sets
Unordered collections of unique elements.
 Example:

  s = {1, 2, 3, 3}
  print(s)  # Outputs {1, 2, 3}
  
13. Tuple Unpacking
Assign multiple values at once.
 Example:

  a, b = 1, 2
  
14. Lambda Functions
Anonymous functions for short, throwaway operations.
 Example: `square = lambda x: x ** 2`

'''


import pandas as pd

##Problem: Write a function reverse_string(s) that takes a string s as input and returns the string reversed.
s = 'hello world'

def rev_string(s):
    return s[::-1]

rev_string(s)
#'dlrow olleh'

#Problem: Write a function is_palindrome(s) that checks if a given string s is a palindrome.
#A palindrome is a word that reads the same backward as forward.

s = 'a man a plan a canal panama'

def is_palindrome(string):
    if string == string[::-1]:
        return(True)
    else:
        return(False)
    
### ALSO def is_palindrome(s):
    ### return s == s[::-1]    

is_palindrome(s)
# 'racecare'  True, 'hello' False
is_palindrome("A man a plan a canal Panama".replace(" ", "").lower())  ##True

##Problem: Write a function fibonacci(n) that returns the nth number in the Fibonacci sequence. 
#The Fibonacci sequence is defined as: F(n) = F(n-1) + F(n-2) for n > 1      [1, 1, 2, 3, 5, 8, etm]

def fibonacci(n):
    if n<= 0 :
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2) 

fibonacci(6)  #8

''' with memoization for optimization 
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
        return memo[n]

'''

##Problem: Write a function max_product(nums) that takes a list of integers and returns the 
##  maximum product of any two numbers.
nums = [5, 20, 2, 6]

def max_product(nums):
    max_prod = float('-inf')  # Set initial max product to negative infinity
    for i in range(len(nums)):  # Loop through each element
        for j in range(i + 1, len(nums)):  # Compare it with every other element
            max_prod = max(max_prod, nums[i] * nums[j])  # Update max product if a larger one is found
    return max_prod

'''
max_prod = float('-inf'):
This initializes max_prod to negative infinity, so any product will be larger than this starting value.

Nested Loops:
The first loop (for i in range(len(nums))) goes through each element in the list.
The second loop (for j in range(i + 1, len(nums))) compares the current element with every other element that comes after it in the list.
For each pair (nums[i], nums[j]), you calculate their product and update max_prod if this product is larger than the current max_prod.

Return:
Finally, return max_prod, which holds the maximum product found.
'''

max_product(nums) ##120
max_product([ 1, 3, -5, -2, 10]) #30

##OPTIMIZED 
def max_products(nums):
    nums.sort()
    return max(nums[-1] * nums[-2], nums[0] * nums[1])

max_products(nums) ##120
max_products([ 1, 3, 5, -2, 10]) #50

##Problem: Write a function merge_intervals(intervals) that takes a list of intervals and merges any 
#   overlapping intervals.

intervals = [(1, 3), (2, 4), (5, 7), (6, 8)]

def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])  # Sort intervals by their starting point
    merged = [intervals[0]]  # Start with the first interval
    
    for current in intervals[1:]:  # Loop through the remaining intervals
        last = merged[-1]  # Get the last added interval in the merged list
        if current[0] <= last[1]:  # If the current interval overlaps with the last one
            merged[-1] = (last[0], max(last[1], current[1]))  # Merge them
        else:
            merged.append(current)  # Otherwise, add the current interval as is
    
    return merged
    
'''
Sorting the Intervals:
intervals.sort(key=lambda x: x[0]): This sorts the intervals based on their starting point. 
Sorting makes it easier to compare and merge intervals.

Merging Logic:
merged = [intervals[0]]: Start by adding the first interval to the merged list.
Loop through the remaining intervals (for current in intervals[1:]):
last = merged[-1]: Get the last interval in the merged list.
if current[0] <= last[1]: Check if the current interval overlaps with the last merged interval.
If they overlap, update the last merged interval to cover both.
merged[-1] = (last[0], max(last[1], current[1])): Merge the intervals by taking the earliest start time and the latest end time.
If they don't overlap, just add the current interval to the merged list.

Return:
Return the merged list, which now contains all the non-overlapping intervals.
'''
merge_intervals(intervals)

##Task: Sort the list of tuples [(1, 3), (2, 1), (4, 2)] by the second element.
tuples = [(1, 3), (2, 1), (4, 2)]
sorted(tuples, key=lambda x: x[1])
#[(2, 1), (4, 2), (1, 3)]

###==========================================

##Task: Write a function sum_two_numbers(a, b) that takes two numbers and returns their sum.
def sum_two_nums(x,y):
    return(x + y) 

sum_two_nums(5,10)
#15

#Task: Write a function max_of_three(a, b, c) that returns the largest of three numbers.
def max_of_three(a,b,c):
    return(max(a,b,c))

max_of_three(100,24,37)
#100

##Task: Write a function count_vowels(s) that counts the number of vowels in a given string.
def count_vowels(string):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    return{char: string.count(char) for char in string.lower() if char in vowels}

count_vowels('Game of Thrones')
##{'a': 1, 'e': 2, 'o': 2}

'''
ALSO 
def count_vowels(s)
    vowels = 'aeiou'
    count = 0
    for char in s:
        if char.lower() in vowels:
            count+=1
    return count

count_vowels('hello world') ##3
'''

##Task: Write a function factorial(n) that returns the factorial of a given number n.
def factorial(n):
    result = 1  # Start with 1, the multiplicative identity
    for i in range(1, n + 1):  # Loop from 1 to n (inclusive)
        result *= i  # Multiply result by the current number i (result = result * i)
    return result  # Return the final factorial value
    
factorial(6)  #720

##Task: Write a function reverse_list(lst) that takes a list and returns it reversed.
def reverse_list(lst): 
    return lst[::-1]
    

reverse_list([1, 2, 3, 4])      #[4, 3, 2, 1]
reverse_list(['a', 'b', 'c'])   #['c', 'b', 'a']


##Task: Write a function is_prime(n) that checks if a number n is prime. A prime number is only divisible by 1 and itself.
import math

def is_prime(n):
    if n <= 1:  # Handle numbers 1 or less (not prime)
        return False
    for i in range(2, int(math.sqrt(n)) + 1):  # Check divisibility up to sqrt(n) (only need to check divisibility up to the square root of n)
        if n % i == 0:  # If n is divisible by i, it's not prime
            return False
    return True  # If no divisors are found, n is prime

is_prime(4)  #False     7  True      3  True        10  False


##Task: Write a function merge_sorted_lists(lst1, lst2) that merges two sorted lists into one sorted list.
def merge_sorted_lists(lst1, lst2):
    merged_list = []  # Initialize an empty list to hold the merged result
    i = j = 0  # Start both pointers at the beginning of their respective lists
    
    # Compare elements in both lists until one is exhausted
    while i < len(lst1) and j < len(lst2):
        if lst1[i] < lst2[j]:  # If element in lst1 is smaller, add it to the merged list
            merged_list.append(lst1[i])
            i += 1  # Move the pointer in lst1 forward
        else:
            merged_list.append(lst2[j])  # Otherwise, add the element from lst2
            j += 1  # Move the pointer in lst2 forward
    
    # Add remaining elements from the list that still has elements
    merged_list.extend(lst1[i:])  # Extend with remaining elements from lst1
    merged_list.extend(lst2[j:])  # Extend with remaining elements from lst2
    
    return merged_list  # Return the merged sorted list


merge_sorted_lists([1, 3, 5], [2, 4, 6])        # [1, 2, 3, 4, 5, 6]
merge_sorted_lists([0, 10, 20], [5, 15, 25])    # [0, 5, 10, 15, 20, 25]


##Task: Write a function intersection(lst1, lst2) that finds the intersection of two lists (i.e., elements that appear in both lists).
def intersection(lst1, lst2):
    return[x for x in lst1 if x in lst2]

intersection([1, 2, 3, 4], [3, 4, 5, 6])  # Output: [3, 4]
intersection(['a', 'b', 'c'], ['c', 'd', 'e'])  # Output: ['c']

###==========================================================
## Loops 
#Task: Write a for loop that prints the numbers from 1 to 5.

for i in range(5):
    print(i+1)
#1
#2
#3
#4
#5

##ALSO for i in range(1,6)
#       print(i)

##Task: Write a for loop that sums all the numbers in the list [1, 2, 3, 4, 5].
numbers = [1, 2, 3, 4, 5]
total = 0 

for num in numbers:
    total += num
print(total)

#15

##Task: Write a for loop that prints only the even numbers from the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for num in numbers:
    if num % 2 == 0:
        print(num)
# 2
# 4
# 6
# 8
# 10

##Task: Write a for loop that counts the number of times the letter 'a' appears in the string "banana".

word = 'banana'
count = 0

for char in word:
    if char == 'a':
        count += 1
print(count)        
    ## 3

##Task: Write a for loop that reverses the string "hello".

word = 'hello'
rev_word = '' 
for char in word: 
    rev_word = char + rev_word
print(rev_word)
##'olleh' 

##Task: Write a program using nested for loops to print the multiplication table from 1 to 3.
for i in range(1,4): 
    for j in range(1,4):
        print(f"{i} x {j} = {i * j}")

# 1 x 1 = 1
# 1 x 2 = 2
# 1 x 3 = 3
# 2 x 1 = 2
# 2 x 2 = 4
# 2 x 3 = 6
# 3 x 1 = 3
# 3 x 2 = 6
# 3 x 3 = 9

##Task: Write a for loop that prints numbers from 1 to 10, but skips the number 5 and stops the loop when it reaches 8.
for i in range (1,11):
    if i == 5:
        continue
    if i == 8:
        break
    print(i)

# 1
# 2
# 3
# 4
# 6
# 7

#Task: Write a for loop to print all the keys and values in the dictionary {'a': 1, 'b': 2, 'c': 3}.
d = {'a': 1, 'b': 2, 'c': 3}

for key, value in d.items():
    print(f"{key}: {value}")

# a: 1
# b: 2
# c: 3

#Task: Write a for loop to print the sum of the elements in each tuple in the list [(1, 2), (3, 4), (5, 6)].

tuples = [(1, 2), (3, 4), (5, 6)]
for a, b in tuples:
    print(a+b)

#Task: Write a while loop that prints numbers from 1 to 5.


num = 1
while  num <6: 
    print(num) 
    num += 1

# 1
# 2
# 3
# 4
# 5
