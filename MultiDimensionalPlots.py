
'''
2019/9/4
By:shenqiti
2D~6D plot
'''
# 2D
import numpy as np
import seaborn as sns
#Read cars data from csv
import pandas as pd
data = pd.read_csv("cars.csv")

#Import modules
import plotly
import plotly.graph_objs as go

#Make Plotly figure
fig1 = go.Scatter(x=data['curb-weight'],
                  y=data['price'],
                  mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(xaxis=dict(title="curb-weight"),
                     yaxis=dict( title="price"))

#Plot and save html
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True)


#4D


import pandas as pd
import plotly
import plotly.graph_objs as go


#Read cars data from csv
data = pd.read_csv("cars.csv")

#Set marker properties
markercolor = data['city-mpg']

#Make Plotly figure
fig1 = go.Scatter3d(x=data['curb-weight'],
                    y=data['horsepower'],
                    z=data['price'],
                    marker=dict(color=markercolor,
                                opacity=1,
                                reversescale=True,
                                colorscale='Blues',
                                size=5),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="curb-weight"),
                                yaxis=dict( title="horsepower"),
                                zaxis=dict(title="price")),)

#Plot and save html
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("4DPlot.html"))


#5D


import pandas as pd
import plotly
import plotly.graph_objs as go


#Read cars data from csv
data = pd.read_csv("cars.csv")

#Set marker properties
markersize = data['engine-size']/12
markercolor = data['city-mpg']

#Make Plotly figure
fig1 = go.Scatter3d(x=data['curb-weight'],
                    y=data['horsepower'],
                    z=data['price'],
                    marker=dict(size=markersize,
                                color=markercolor,
                                opacity=0.9,
                                reversescale=True,
                                colorscale='Blues'),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="curb-weight"),
                                yaxis=dict( title="horsepower"),
                                zaxis=dict(title="price")),)

#Plot and save html
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("5D Plot.html"))






#6D


import pandas as pd
import plotly
import plotly.graph_objs as go


#Read cars data from csv
data = pd.read_csv("cars.csv", error_bad_lines=False)

#Set marker properties
markersize = data['engine-size']/12
markercolor = data['city-mpg']
markershape = data['num-of-doors'].replace("four","square").replace("two","circle")


#Make Plotly figure
fig1 = go.Scatter3d(x=data['curb-weight'],
                    y=data['horsepower'],
                    z=data['price'],
                    marker=dict(size=markersize,
                                color=markercolor,
                                symbol=markershape,
                                opacity=0.9,
                                reversescale=True,
                                colorscale='Blues'),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="curb-weight"),
                                yaxis=dict( title="horsepower"),
                                zaxis=dict(title="price")),)

#Plot and save html
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("6DPlot.html"))










