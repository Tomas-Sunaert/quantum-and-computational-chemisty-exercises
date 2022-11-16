import numpy as np 
import plotly.graph_objects as go

E= np.array([-108.970423343, -108.971565380, -108.971947136,-108.971617061,-108.970662031,-108.969002005])
R= np.array([1.05, 1.06,1.07,1.08,1.09,1.1])
aaaa= np.arange(1.04,1.11,0.001)
fit = np.polyfit(R,E, 2)
f = np.poly1d(fit)
y= f(aaaa)
fig = go.Figure()
fig.add_trace(go.Scatter(x=R, y=E ,mode = 'markers'))
fig.add_trace(go.Scatter(x=aaaa, y =y, mode = 'lines'))
fig.show()
print(fit)
