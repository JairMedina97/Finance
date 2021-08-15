import matplotlib.pyplot as plt

# Data to plot
labels = 'Online stores', 'Third-party seller services','Amazon Web Services' , 'Physical stores', 'Subscription services', 'Other' 
sizes = [122987, 42745, 25655, 17224 , 14168, 10108]
colors = ['darkgoldenrod', 'yellow', 'black', 'khaki', 'gold', 'dimgray']
explode = (0, 0, 0, 0, 0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', startangle=270)

plt.axis('equal')
plt.title('Amazon Revenue Breakdown')

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)


plt.show()
