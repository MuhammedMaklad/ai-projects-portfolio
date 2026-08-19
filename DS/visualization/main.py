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

# %%
import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
from holoviews.examples.gallery.apps.bokeh.game_of_life import title
from matplotlib.pyplot import xticks

# %%
df = pd.read_csv("Canada.csv")

# %%
df.head()

# %%
df.set_index("Country",inplace=True)    

# %%
df.head()

# %%
df.index.name = 'Country'

# %%
df.head()

# %%
years = list(map(str, range(1980, 2014)))

# %%
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
haiti = df.loc['Haiti',years]

# %%
print(haiti.head())

# %%
haiti.plot(kind='line',color='blue')
plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.show()

# %%
haiti.index = haiti.index.map(int)
haiti.plot(kind='line',color='blue')    

plt.title("Immigration from Haiti")
plt.ylabel("Number of Immigrations")
plt.xlabel("Years")

plt.text(2000, 6000, "2010 Earthquake")
plt.show()

# %%
from plotly import  __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

# %%
print(__version__)

# %%
import cufflinks as cf

# %%
# For Notebooks
init_notebook_mode(connected=True)

# %%
# For offline use
cf.go_offline()

# %%
# Assuming you have already defined `haiti` as a pandas Series containing the data

# Convert index to integers
haiti.index = haiti.index.map(int)

# Create a scatter plot using Cufflinks
fig = haiti.iplot(kind='scatter', mode='lines+markers', title='Immigration from Haiti',
                  xTitle='Years', yTitle='Number of Immigrants', dimensions=(800, 500), asFigure=True)

# Add annotation for the 2010 Earthquake
fig.add_trace(go.Scatter(x=[2010], y=[haiti[2010]], mode='markers', name='2010 Earthquake',
                         marker=dict(color='red', size=10), showlegend=False,
                         text=['2010 Earthquake'], textposition='bottom center'))

# Show the plot
fig.show()


# %%
df.sort_values(['Total'],ascending=False, axis = 0, inplace = True)

# get the top 5 entries
df_top5 = df.head()

# transpose the dataframe
df_top5 = df_top5[years].transpose()

df_top5.head()

# %%
# let's change the index values of df_top5 to type of integer for plotting 
df_top5.index = df_top5.index.map(int)

df_top5.plot(kind='area', 
             stacked=False,
             figsize=(20, 10),
             )

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()

# %%
# option 2: preferred option with more flexibility
ax = df_top5.plot(kind='area', alpha=0.35, figsize=(20, 10))

ax.set_title('Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')

# %% [markdown]
# ## Using plotly and cufflinks

# %%
# Assuming you have already defined 'df_top5' as a DataFrame containing the top 5 countries immigration data 

# Create a Plotly figure using Cufflinks
fig = df_top5.iplot(kind='area', 
                        xTitle='Years', 
                        yTitle='Number of Immigrants', 
                        title='Immigration Trend of Top 5 Countries',
                        colorscale='set1',  # Set the colorscale
                        fill=True,  # Enable area filling
                        opacity=0.5,  # Set opacity
                        theme='pearl',  # Set plot theme
                        dimensions=(900, 500)  # Set plot dimensions
)

# %% [markdown]
# ## Histograms

# %%
# let's quickly view the 2013 data
df['2013'].head()

# %%
# np.histogram returns 2 values
count, bin_edges = np.histogram(df['2013'], 15)

print(count) # frequency count
print(bin_edges) # bin ranges, default = 10 bins

    # %%
    df['2013'].plot(kind='hist', figsize=(8, 5))
    plt.title('Histogram of Immigration from 195 countries in 2013')
    plt.ylabel('Number of Countries')
    plt.xlabel('Number of Immigrants')
    plt.show()

# %%
# bin_edges is a list of bin intervals

count, bin_edges = np.histogram(df['2013'])

df['2013'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)

plt.title('Histogram of Immigration from 195 countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')

plt.show()


# %%
df['2013'].plot.hist(xticks=bin_edges)
plt.title('Histogram of Immigration from 195 countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')
plt.show()

# %%
histogram = df['2013'].iplot(kind='histogram',bins = len(bin_edges), xTitle="Number of Immigrants", yTitle="Number of Countries", title='Histogram of Immigration from 195 countries in 2013', dimensions=(800, 500), asFigure=True)

# update layout
histogram.update_layout(showlegend=False)
histogram.show()

# %%
df_t = df.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()

# Un-stacked histogram with custom size
df_t.iplot(kind='hist', 
          bins=15,
          color=['coral', 'darkslateblue', 'mediumseagreen'],
          layout={
              'title': 'Histogram of Values',  # Optional title
              'width': 800,                   # Width in pixels
              'height': 500,                 # Height in pixels
              'bargap': 0.05,                   # Gap between bars (optional)
          })

# %%
# step 1: get the data
df_iceland = df.loc['Iceland',years]
df_iceland.head()

# %%
# step 2: plot data
df_iceland.plot(kind='bar', figsize=(10, 6), rot=90)

plt.title('Icelandic Immigration to Canada from 1980 to 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.show()

# %%
df_iceland.plot.bar(figsize=(10,6), rot=90)
plt.title('Icelandic Immigration to Canada from 1980 to 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

# Annotate arrow
plt.annotate('', 
             xy=(32, 70), 
             xytext=(28, 20), 
             xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='green', lw=2)
             )
plt.show()

# %%
df_iceland.plot(kind='bar', figsize=(10, 6), rot=90)

plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.title('Icelandic Immigrants to Canada from 1980 to 2013')

# Annotate arrow
plt.annotate('',  # s: str. will leave it blank for no text
             xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',  # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
             )

# Annotate Text
plt.annotate('2008 - 2011 Financial Crisis',  # text to display
             xy=(28, 30),  # start the text at at point (year 2008 , pop 30)
             rotation=75,  # based on trial and error to match the arrow
             va='bottom',  # want the text to be vertically 'bottom' aligned
             ha='left',  # want the text to be horizontally 'left' algned.
             )

plt.show()

# %%
# creating a bar using Cufflinks
bar_plot = df_iceland.iplot(
    kind='bar',
    title='Icelandic Immigration to Canada from 1980 to 2013',
    xTitle='Year',
    yTitle='Number of Immigrants',
    dimensions=(800, 500),
    asFigure=True
)

# Add arrow annotation
bar_plot.add_annotation(
    x=32,y=70, xref='x', yref='y',
    ax=28, ay = 20,
    showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2,
    arrowcolor='blue',
    axref='x',ayref='y'
)

# Add text annotation
bar_plot.add_annotation(x=30, y=40, xref="x", yref="y",
                        text="2008 - 2011 Financial Crisis",
                        textangle=-72.5, align="center")

# Update layout
#bar_plot.update_layout(xaxis_tickangle=-90, showlegend=False)

# Show the plot
bar_plot.show()

# %%
print(7 * 19)

# %%
df_continents = df.groupby('Continent',axis=0).sum()
df_continents['Total'].head()

# %%
df_continents['Total'].plot(kind='pie',
                            figsize=(5,6),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            labels=None,
                            pctdistance=1.12,
                            )

plt.title('Immigration to Canada by Continent [1980-2013]')
plt.axis('equal')
plt.legend(labels=df_continents.index, loc='upper left')
plt.show()

# %%
colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.

df_continents['Total'].plot(kind='pie',
                            figsize=(10, 6),
                            autopct='%1.1f%%', 
                            startangle=90,    
                            shadow=True,       
                            labels=None,         # turn off labels on pie chart
                            pctdistance=1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
                            colors=colors_list,  # add custom colors
                            explode=explode_list # 'explode' lowest 3 continents
                            )

# scale the title up by 12% to match pctdistance
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12, fontsize = 15) 

plt.axis('equal') 

# add legend
plt.legend(labels=df_continents.index, loc='upper left', fontsize=7) 

plt.show()

# %%
import plotly.express as px

# %%
random_x = df_continents['Total']
names = df_continents.index.to_list()

fig = px.pie(values=random_x, names=names,title="Immigration to Canada by Continent [1980 - 2013]")
fig.show()

# %%
df_japan = df.loc[['Japan'],years].transpose()
df_japan 

# %%
df_japan.plot(kind='box',figsize=(8, 6))
plt.title('Box plot of Japanese Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')
plt.show()

# %%
box_plot = df_japan.iplot(kind='box',
                          title="Box plot of Immigration from 1980 - 2013",
                          yTitle="Number of Immigrants",
                          asFigure=True
                          )
box_plot.show()

# %%
