def clear():
    print('\n' * 40)
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pylab
import matplotlib.ticker as ticker

california = np.array([1378653,
1470393,
1582377,
1709938,
1702775,
1743650,
1825424,
1902318,
1990140,
2072177,
2003618,
2011138,
2076486,
2258137,
2391586,
2444497,
2520867,
2609927,
2726142,
2898839,
2987571,
3077939,
])

canada = np.array([652823,
631813,
676084,
742295,
736379,
757950,
892382,
1023196,
1169357,
1315415,
1464977,
1549131,
1371153,
1613542,
1789140,
1823966,
1842018,
1801480,
1552899,
1526705,
1646867,
1709327,
])

uk = np.array([1652823,
1731813,
1776084,
1642295,
1622379,
1768950,
2038000,
2399196,
2521357,
2693415,
2874977,
2791131,
2381153,
2443542,
2629140,
2662966,
2742018,
2923000,
2886000,
2751000,
2622000,
2709327,
])

plt.style.use('ggplot')

n = 8

x = range(1997,2019)


fig, ax = plt.subplots()

pylab.plot(x, california, 'g', label='California')
pylab.plot(x, canada, 'r', label='Canada')
pylab.plot(x, uk, 'b', label='United Kingdom')

plt.title("Gross Domestic Product", fontsize=18, y=1)
plt.xlabel("Date", fontsize=14, labelpad=15)
plt.ylabel("US Million Dollars", fontsize=14, labelpad=15)



#for i in range(n):
#    pylab.axhline(0 + i, color='gray', linewidth=1)


plt.axvspan(2007, 2009, facecolor='gray', alpha=0.4,zorder=3)

pylab.legend(loc='lower right')

# place a text box in upper left in axes coords

textstr = '\n'.join((
    r'$\mathrm{JAIR MEDINA}%.2f$',))

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='deepskyblue', alpha=0.9)

ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

formatter = ticker.FormatStrFormatter('$%1.2f')
ax.yaxis.set_major_formatter(formatter)

for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_visible(False)
    tick.label2.set_visible(True)
    #tick.label2.set_color('green')


pylab.show()
