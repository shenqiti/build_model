import pandas as pd
import plotly
import plotly.graph_objs as go
import numpy as np

#Read cars data from csv
data = pd.read_csv("./data_finall/day_data/hour2day_sort.csv")

#Set marker properties
markersize =  data['PRES']/ 75
markercolor = data['Iws']
Is = data['Is']
Is1 = []


for i in range(0,len(Is)):
    if Is[i] != 0:
        Is1.append("circle")
    else:
        Is1.append("square")


markershape = Is1

x = data['DEWP']
x = np.array(x)
x1 = []
y1 = []
z1 = []

y = data['TEMP']
y = np.array(y)
z = data['pm2.5']
z = np.array(z)

for each in x:
    x1.append(int(each))
for each in y:
    y1.append(int(each))
for each in z:
    z1.append(int(each))





#Make Plotly figure
fig1 = go.Scatter3d(x=x1,
                    y=y1,
                    z=z1,
                    marker=dict(size=markersize,
                                color=markercolor,
                                symbol=markershape,
                                opacity=0.9,
                                reversescale=True,
                                colorscale='Blues'),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="DEWP"),
                                yaxis=dict( title="TEMP"),
                                zaxis=dict(title="pm2.5")),)

#Plot and save html
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("6DPlot.html"))






