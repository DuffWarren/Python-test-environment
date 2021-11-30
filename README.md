# Python-test-environment

# Visualizations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

# Global API
Matplotlib's default pyplot API has a global, MATLAB style interface

x = np.arange(-10, 11)

plt.figure(figsize=(12, 6))

plt.title('My Nice Plot')

plt.plot(x, x ** 2)
plt.plot(x, -1 * (x **2))
# (x, x^2), vs. (x, -x^2)

plt.figure(figsize=(12, 6))
plt.title('My Nice Plot')

plt.subplot(1, 2, 1) # Rows, Columns, Panel Selected
plt.plot(x, x ** 2) # This is the plot
plt.plot([0, 0, 0], [-10, 0, 100]) # This is the vertical line
# x = 0 at y = -10 (line goes below 0 value on plot), 0, and 100
plt.legend(['X^2', 'Vertical Line'])
plt.xlabel('X')
plt.ylabel('X Squared')

plt.subplot(1, 2, 2) 
plt.plot(x, -1 * (x **2))
plt.plot([-10, 0, 10], [-50, -50, -50]) # This is the horizontal line
# at x = -10, 0, and 10 y = -50, -50, and -50 to create line
plt.legend(['-X^2', 'Horizontal Line'])
plt.xlabel('X')
plt.ylabel('X Squared')



# OOP Interface
Object-Oriented approach

fig, axes = plt.subplots(figsize=(12, 6))
# note the comma after the 'fig'
# blank plot

axes.plot(
    x, (x **2), color='red', linewidth=3,
    marker='o', markersize=8, label='X^2')
# in this colour is spelled without the 'u'

axes.plot(x, -1 * (x **2), 'b--', label='-X^2')

axes.set_xlabel('X')
axes.set_ylabel('X Squared')

axes.set_title('My Nice Plot')

axes.legend()

fig
# watch the first-line capitalization! 

# Matplotlib Visualizations

fig, axes = plt.subplots(figsize=(12, 6))

axes.plot(x, x + 0, linestyle='solid', color='purple', label='Happy')
axes.plot(x, 2 * (x + 1), linestyle='dashed', label='Sad')
axes.plot(x, 3 * (x + 2), linestyle='dashdot', label='Curious')
axes.plot(x, 4 * (x + 3), linestyle='dotted', label='Anxious');

axes.set_xlabel('X')
axes.set_ylabel('X Squared')

axes.legend()
# this links to the 'label=' in the axes.plot lines

axes.set_title("My Nice Plot")

## Codes to create line type and colour

fig, axes = plt.subplots(figsize=(12, 6))

axes.plot(x, 1 * (x + 0), '-og', label='Solid green with dots')
axes.plot(x, 2 * (x + 1), '--oc', label='Dashed Cyan with dots')
axes.plot(x, 3 * (x + 2), '-.b', label='Dash Dot Blue')
axes.plot(x, 4 * (x + 3), ':r', label='Dotted Red')

axes.set_xlabel('X')
axes.set_ylabel('X Squared')

axes.set_title("My Nice Plot")

axes.legend()

## Setting number of graphs in our ouput

We can do this using the 'nrows' and 'ncols' parameters

plot_objects = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))

fig, ((ax1, ax2), (ax3, ax4)) = plot_objects

plot_objects
# gives the text description below

# this one is weird and I don't know why
ax4.plot(np.random.randn(10), c='yellow')
ax3.plot(np.random.randn(10), c='red', linestyle='--')
ax2.plot(np.random.randn(10), c='green', linestyle=':')
# must use 'marker' in this instance not ':o'
ax1.plot(np.random.randn(10), c='blue', marker='o', linewidth=1.0)

fig

## Subplots using a grid format
'subplot2grid'

'colspan' and 'rowspan'

plt.figure(figsize=(14, 6))

ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
# stretch across 3 columns
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
# stretch across 2 rows
ax4 = plt.subplot2grid((3,3), (2,0))
ax5 = plt.subplot2grid((3,3), (2,1))
# unspecified display in 1 column and 1 row

## Scatter Plots
### Can graph 4 dimensions of data
x, y, size of point and colour intensity of point

N = 80

x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (20 * np.random.rand(N))**2

# numpy randomization
# numpy pie chart area

plt.figure(figsize=(14, 6))

plt.scatter(x, y, s=area, c=colors, alpha=0.8, cmap='Spectral')
plt.colorbar()

plt.show()

# no 'ax_' defined so default one created by default
# 6 arguments for scatter plot
# alpha = colour transparency
# cmap = type of colour scale (see link below)

## Two plots together

plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1,2,1) # 1 Row, 2 Columns, Panel 1 Selected
plt.scatter(x, y, s=area, c=colors, alpha=0.5, cmap='Pastel1')
plt.colorbar()

ax2 = fig.add_subplot(1,2,2) # 1 Row, 2 Columns, Panel 2 Selected
plt.scatter(x, y, s=area, c=colors, alpha=0.5, cmap='Pastel2')
plt.colorbar()

plt.show()

plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1,2,1) # 1 Row, 2 Columns, Panel 1 Selected
plt.scatter(x, y, s=area, c=colors, alpha=0.5, cmap='Pastel1')
plt.colorbar()



plt.show()

### Link to all types of colourmap types 'cmap'
https://matplotlib.org/users/colormaps.html 

Think about printing in black and white as well as choosing either dual plots for colourblindness (this makes it easier to see anyway) or a tab that allows switching in live reports or dashboards
https://www.vischeck.com/ 

## Histograms

values = np.random.randn(1000)

plt.subplots(figsize=(12, 6))

plt.hist(values, bins=100, alpha=0.8,
        histtype='bar', color='steelblue',
        edgecolor='green')
plt.xlim(xmin=-5, xmax=5)

plt.show()

# bins with an 's'

### Saving plots as .png

fig.savefig('hist.png')

# Nothing happened? Does not work in a notebook?

## KDE (kernal density estimation)

from scipy import stats

density = stats.kde.gaussian_kde(values)
density

plt.subplots(figsize=(12, 6))

values2 = np.linspace(min(values)-10, max(values)+10, 100)

plt.plot(values2, density(values2), color='#FF7F00')
plt.fill_between(values2, 0, density(values2), alpha=0.5, color='#FF7F00')
plt.xlim(xmin=-5, xmax=+5)

plt.show()

# linspace not linEspace

## Combining plots

plt.subplots(figsize=(12, 6))

plt.hist(values, bins=100, alpha=0.5, density=1,
        histtype='bar', color='steelblue',
        edgecolor='purple')

plt.plot(values2, density(values2), color='#FF7F00')
plt.xlim(xmin=-2, xmax=2)

plt.show()

# density added to hisogram - comma to separate values
# 'plt.plot' is only plotting the line without the fill as above

## Boxplots 

Y = np.random.rand(1, 5)[0]
Y2 = np.random.rand(1, 5)[0]

# Set variables first

plt.subplots(figsize=(12, 6))

barWidth = 0.5
plt.bar(np.arange(len(Y)), Y, width=barWidth, color='#00b894', label='Label Y')

plt.show()

## Stacked Boxplots

plt.subplots(figsize=(12, 6))

barWidth = 0.5
plt.bar(np.arange(len(Y)), Y, width=barWidth, color='#00b894', label='Label Y')
plt.bar(np.arange(len(Y2)), Y2, width=barWidth, color='#e17055', bottom=Y, 
        label='Label Y2')

plt.legend()
plt.show()

# automatically adjusts legend up from 1 to 2.0 to accomaodate stacked values

## Boxplots and outlier detection

values = np.concatenate([np.random.randn(10), np.array([10, 15, -10, -15])])

# 10 randomized values
# not sure how this randomly generated values

df = pd.DataFrame({
    'array1': [1., 1., 0., 0., 5., 5., 0., 0., 1., 1.],
    'array2': [-15., -12.,  -9.,  -6.,  -3.,   3.,   6.,   9.,  12.,  15.]
})
df

# took array data from below
# had to delete middle value from second array to display properly

plt.figure(figsize=(12, 4))

plt.hist(values)

plt.figure(figsize=(12, 4))

plt.boxplot(values)

# mean = 0 (values are equally distributed on both sides) 
# why does this show a skew to the positive values? 







