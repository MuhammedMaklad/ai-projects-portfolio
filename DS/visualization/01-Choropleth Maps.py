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
# ___
#
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# %% [markdown]
# # Choropleth Maps

# %% [markdown]
# ## Offline Plotly Usage

# %% [markdown]
# Get imports and set everything up to be working offline.

# %%
#import plotly.plotly as py
#import plotly.graph_objs as go 
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# %% [markdown]
# Now set up everything so that the figures show up in the notebook:

# %%
init_notebook_mode(connected=True) 

# %% [markdown]
# More info on other options for Offline Plotly usage can be found [here](https://plot.ly/python/offline/).

# %% [markdown]
# ## Choropleth US Maps
#
# Plotly's mapping can be a bit hard to get used to at first, remember to reference the cheat sheet in the data visualization folder, or [find it online here](https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf).

# %%
import pandas as pd

# %% [markdown]
# Now we need to begin to build our data dictionary. Easiest way to do this is to use the **dict()** function of the general form:
#
# * type = 'choropleth',
# * locations = list of states
# * locationmode = 'USA-states'
# * colorscale= 
#
# Either a predefined string:
#
#     'pairs' | 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' | 'Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
#
# or create a [custom colorscale](https://plot.ly/python/heatmap-and-contour-colorscales/)
#
# * text= list or array of text to display per point
# * z= array of values on z axis (color of state)
# * colorbar = {'title':'Colorbar Title'})
#
# Here is a simple example:

# %%
data = dict(type = 'choropleth',
            locations = ['AZ','CA','NY'],
            locationmode = 'USA-states',
            colorscale= 'Portland',
            text= ['text1','text2','text3'],
            z=[1.0,2.0,3.0],
            colorbar = {'title':'Colorbar Title'})

# %% [markdown]
# Then we create the layout nested dictionary:

# %%
layout = dict(geo = {'scope':'usa'})

# %% [markdown]
# Then we use: 
#
#     go.Figure(data = [data],layout = layout)
#     
# to set up the object that finally gets passed into iplot()

# %%
choromap = go.Figure(data = [data],layout = layout)

# %%
iplot(choromap)

# %% [markdown]
# ### Real Data US Map Choropleth
#
# Now let's show an example with some real data as well as some other options we can add to the dictionaries in data and layout.

# %%
df = pd.read_csv('2011_US_AGRI_Exports')
df.head()

# %% [markdown]
# Now out data dictionary with some extra marker and colorbar arguments:

# %%
data = dict(type='choropleth',
            colorscale = 'Viridis',
            locations = df['code'],
            z = df['total exports'],
            locationmode = 'USA-states',
            text = df['text'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Millions USD"}
            ) 

# %% [markdown]
# And our layout dictionary with some more arguments:

# %%
layout = dict(title = '2011 US Agriculture Exports by State',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )

# %%
choromap = go.Figure(data = [data],layout = layout)

# %%
iplot(choromap)

# %% [markdown]
# # World Choropleth Map
#
# Now let's see an example with a World Map:

# %%
df = pd.read_csv('2014_World_GDP')
df.head()

# %%
data = dict(
        type = 'choropleth',
        locations = df['CODE'],
        z = df['GDP (BILLIONS)'],
        text = df['COUNTRY'],
        colorbar = {'title' : 'GDP Billions US'},
      ) 

# %%
layout = dict(
    title = '2014 Global GDP',
    geo = dict(
        showframe = False,
        projection = {'type':'natural earth'}
    )
)

# %%
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)

# %% [markdown]
# # Great Job!
