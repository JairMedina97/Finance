import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import plotly.graph_objects as go


states = ['Aguascalientes',
'Baja California',
'Baja California Sur',
'Campeche',
'Chiapas',
'Chihuahua',
'Ciudad de México',
'Coahuila de Zaragoza',
'Colima',
'Durango',
'Estado de México',
'Guanajuato',
'Guerrero',
'Hidalgo',
'Jalisco',
'Michoacán de Ocampo',
'Morelos',
'Nayarit',
'Nuevo León',
'Oaxaca',
'Puebla',
'Querétaro',
'Quintana Roo',
'San Luis Potosí',
'Sinaloa',
'Sonora',
'Tabasco',
'Tamaulipas',
'Tlaxcala',
'Veracruz de Ignacio de la Llave',
'Yucatán',
'Zacatecas']

a = np.array([1312544,
3315766,
712029,
899931,
2954915,
3556574,
8918653,
2954915,
711235,
1754754,
16187608,
5853677,
3533251,
2858359,
7844830,
4584471,
1903811,
1181050,
5119504,
3967889,
6168883,
2038372,
1501562,
2717820,
2966321,
2850330,
2395272,
3441698,
1272847,
8112505,
2097175,
1579209])

poblacion = a.reshape(len(a),1)

b = np.array([0.282,
0.222,
0.221,
0.436,
0.248,
0.336,
0.771,
0.306,
0.276,
0.36,
0.424,
0.644,
0.506,
0.318,
0.478,
0.553,
0.495,
0.375,
0.142,
0.704,
0.594,
0.311,
0.288,
0.455,
0.308,
0.279,
0.509,
0.322,
0.539,
0.622,
0.419,
0.49])

povertyrate = b.reshape(len(b),1)

c = numpy.multiply(a,b)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
               fill_color='lavender',
               align='left'))
])

fig.show()

# create data
#s = [1.2**n for n in range(len(x))]

#use the scatter function
#plt.scatter(x, y, z, alpha=0.5)


N = 32
M = 4 # Number of bins
 
x = b 
y = a / 1000
a2 = c / 10000
 
# Create the DataFrame from your randomised data and bin it using groupby.
df = pd.DataFrame(data=dict(x=x, y=y, a2=a2))
bins = np.linspace(df.a2.min(), df.a2.max(), M)
grouped = df.groupby(np.digitize(df.a2, bins))
 
# Create some sizes and some labels.
sizes = [32*(i+1.) for i in range(M)]
labels = ['Norte', 'Centro', 'Sur', 'Este']
 
for i, (name, group) in enumerate(grouped):
    plt.scatter(group.x, group.y, s=sizes[i], alpha=0.5, label=labels[i])
 
plt.legend()
plt.show()



 






