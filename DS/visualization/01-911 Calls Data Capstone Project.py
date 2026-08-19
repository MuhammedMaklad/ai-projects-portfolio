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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 911 Calls Capstone Project

# %% [markdown]
# For this capstone project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
#
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
#
# Just go along with this notebook and try to complete the instructions or answer the questions in bold using your Python and Data Science skills!

# %% [markdown]
# ## Data and Setup

# %% [markdown]
# ____
# ** Import numpy and pandas **

# %%

# %% [markdown]
# ** Import visualization libraries and set %matplotlib inline. **

# %%

# %% [markdown]
# ** Read in the csv file as a dataframe called df **

# %%

# %% [markdown]
# ** Check the info() of the df **

# %%

# %% [markdown]
# ** Check the head of df **

# %%

# %% [markdown]
# ## Basic Questions

# %% [markdown]
# ** What are the top 5 zipcodes for 911 calls? **

# %%

# %% [markdown]
# ** What are the top 5 townships (twp) for 911 calls? **

# %%

# %% [markdown]
# ** Take a look at the 'title' column, how many unique title codes are there? **

# %%

# %% [markdown]
# ## Creating new features

# %% [markdown]
# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
#
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# %%

# %% [markdown]
# ** What is the most common Reason for a 911 call based off of this new column? **

# %%

# %% [markdown]
# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# %%

# %% [markdown]
# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# %%

# %% [markdown]
# ** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# %%

# %% [markdown]
# ** You can now grab specific attributes from a Datetime object by calling them. For example:**
#
#     time = df['timeStamp'].iloc[0]
#     time.hour
#
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**

# %%

# %% [markdown]
# ** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **
#
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# %%

# %%

# %% [markdown]
# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# %%

# %% [markdown]
# **Now do the same for Month:**

# %%

# %% [markdown]
# **Did you notice something strange about the Plot?**
#
# _____
#
# ** You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas... **

# %% [markdown]
# ** Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **

# %%

# %% [markdown]
# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# %%

# %% [markdown]
# ** Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# %%

# %% [markdown]
# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# %%

# %% [markdown]
# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# %%

# %% [markdown]
# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# %%

# %%

# %%

# %% [markdown]
# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# %%

# %% [markdown]
# ** Now create a HeatMap using this new DataFrame. **

# %%

# %% [markdown]
# ** Now create a clustermap using this DataFrame. **

# %%

# %% [markdown]
# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# %%

# %%

# %%

# %% [markdown]
# **Continue exploring the Data however you see fit!**
# # Great Job!
