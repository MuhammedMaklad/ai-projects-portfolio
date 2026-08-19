# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %% [markdown]
# ___
#
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Python Crash Course Exercises 
#
# This is an optional exercise to test your understanding of Python Basics. If you find this extremely challenging, then you probably are not ready for the rest of this course yet and don't have enough programming experience to continue. I would suggest you take another course more geared towards complete beginners, such as [Complete Python Bootcamp](https://www.udemy.com/complete-python-bootcamp/?couponCode=PY20)

# %% [markdown]
# ## Exercises
#
# Answer the questions or complete the tasks outlined in bold below, use the specific method described if applicable.

# %% [markdown]
# ** What is 7 to the power of 4?**

# %%
print(7**4)

# %% [markdown]
# ** Split this string:**
#
#     s = "Hi there Sam!"
#     
# **into a list. **

# %%
s = "Hi there Sam!"

# %%
list(s.split(" "))

# %% [markdown]
# ** Given the variables:**
#
#     planet = "Earth"
#     diameter = 12742
#
# ** Use .format() to print the following string: **
#
#     The diameter of Earth is 12742 kilometers.

# %%
planet = "Earth"
diameter = 12742

# %%
print("The diameter of {} is {} Kilometers".format(planet,diameter))

# %% [markdown]
# ** Given this nested list, use indexing to grab the word "hello" **

# %%
lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]

# %%
lst[3][1][2][0]

# %% [markdown]
# ** Given this nested dictionary grab the word "hello". Be prepared, this will be annoying/tricky **

# %%
d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}

# %%
d['k1'][3]['tricky'][3]['target'][3]


# %% [markdown]
# ** What is the main difference between a tuple and a list? **

# %%
# Tuple is immutable
# List mutable

# %% [markdown]
# ** Create a function that grabs the email website domain from a string in the form: **
#
#     user@domain.com
#     
# **So for example, passing "user@domain.com" would return: domain.com**

# %%
def domainGet(str):
    return str.split('@')[1]


# %%
domainGet('user@domain.com')


# %% [markdown]
# ** Create a basic function that returns True if the word 'dog' is contained in the input string. Don't worry about edge cases like a punctuation being attached to the word dog, but do account for capitalization. **

# %%
def findDog(s):
    return 'dog' in s.lower()


# %%
findDog('Is there a dog here?')


# %% [markdown]
# ** Create a function that counts the number of times the word "dog" occurs in a string. Again ignore edge cases. **

# %%
def countDog(s):
    return s.split(" ").count("dog")
    # return len(list(filter(lambda x: x.lower() == 'dog', s.split())))


# %%
countDog('This dog runs faster than the other dog dude!')

# %% [markdown]
# ** Use lambda expressions and the filter() function to filter out words from a list that don't start with the letter 's'. For example:**
#
#     seq = ['soup','dog','salad','cat','great']
#
# **should be filtered down to:**
#
#     ['soup','salad']

# %%
seq = ['soup','dog','salad','cat','great']

# %%
list(filter(lambda  x: x[0] =='s',seq))


# %% [markdown]
# ### Final Problem
# **You are driving a little too fast, and a police officer stops you. Write a function
#   to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". 
#   If your speed is 60 or less, the result is "No Ticket". If speed is between 61 
#   and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big    Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all 
#   cases. **

# %%
def caught_speeding(speed, is_birthday):
    added = 5 if is_birthday else 0 
    return "No Ticket" if speed <= 60 + added else "Small Ticket" if 61 + added <= speed <= 80 + added else "Big Ticket"


# %%
caught_speeding(81,True) #'Small Ticket'

# %%
caught_speeding(81,False)

# %% [markdown]
# # Great job!
