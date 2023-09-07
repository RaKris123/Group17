import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# %matplotlib inline

# Import required libraries, matplotlib library for visualizing, and CSV library for reading CSV data.
# Open the file using open( ) function with ‘r’ mode (read-only) from CSV library and read the file using csv.reader( ) function.
# Read each line in the file using for loop.
# Append required columns into a list.
# After reading the whole CSV file, plot the required data as X and Y axis.
# In this example, we are plotting names as X-axis and ages as Y-axis.

#class1 lists for x,y coordinates
xCoord1 = []
yCoord1 = []
#class2 lists for x,y coordinates
xCoord2 = []
yCoord2 = []
#class3 lists for x,y coordinates
xCoord3 = []
yCoord3 = []


#class1 path
path1 = "Class1.csv"
#class2 path
path2 = "Class2.csv"
#class3 path
path3 = "Class3.csv"

#class1 opening file with path in readable mode and giving it as an alias
with open(path1, 'r') as dataset1:
    grid1 = csv.reader(dataset1) #object of csv module reads dataset
    for tupl1 in grid1:           #iteration
        try:                                      #exception handling try and catch
            xCoord1.append(float(tupl1[0]))
            yCoord1.append(float(tupl1[1]))
        except (ValueError, IndexError):
            pass
        
        
x_coord1_sum=0.0
x_coord1_count=0

mean_xCoord1=0

for x_point1 in xCoord1:
    x_coord1_sum = x_coord1_sum + float(x_point1)
    x_coord1_count = x_coord1_count + 1

mean_xCoord1 = x_coord1_sum/x_coord1_count

print('\nMean of X coordinates of class 1 of LS class is: ', mean_xCoord1)

y_coord1_sum=0.0
y_coord1_count=0

mean_yCoord1=0

for y_point1 in yCoord1:
    y_coord1_sum = y_coord1_sum + float(y_point1)
    y_coord1_count = y_coord1_count + 1

mean_yCoord1 = y_coord1_sum/y_coord1_count

print('\nMean of Y coordinates of class 1 of LS class is: ', mean_yCoord1)


datavector1 = []
for i in range(x_coord1_count):
    datavector1.append((xCoord1[i],yCoord1[i]))
    

cov_sum1=0.0
for x,y in datavector1:
    cov_sum1 = cov_sum1 + (x - mean_xCoord1)*(y - mean_yCoord1)/x_coord1_count


print('\nCovariance of class 1 of LS class1 is:', cov_sum1)


#class2
with open(path2, 'r') as dataset2:
    grid2 = csv.reader(dataset2)
    for tupl2 in grid2:
        try:
            xCoord2.append(float(tupl2[0]))
            yCoord2.append(float(tupl2[1]))
        except (ValueError, IndexError): 
            pass
        
        
x_coord2_sum=0.0
x_coord2_count=0

mean_xCoord2=0

for x_point2 in xCoord2:
    x_coord2_sum = x_coord2_sum + float(x_point2)
    x_coord2_count = x_coord2_count + 1

mean_xCoord2 = x_coord2_sum/x_coord2_count

print('\nMean of X coordinates of class 2 of LS class is: ', mean_xCoord2)

y_coord2_sum=0.0
y_coord2_count=0

mean_yCoord2=0

for y_point2 in yCoord2:
    y_coord2_sum = y_coord2_sum + float(y_point2)
    y_coord2_count = y_coord2_count + 1

mean_yCoord2 = y_coord2_sum/y_coord2_count

print('\nMean of Y coordinates of class 2 of LS class is: ', mean_yCoord2)
      
#class3        
with open(path3, 'r') as dataset3: 
    grid3 = csv.reader(dataset3)
    for tupl3 in grid3:
        try: 
            xCoord3.append(float(tupl3[0]))
            yCoord3.append(float(tupl3[1]))
        except (ValueError, IndexError):
            pass
        
        
        
x_coord3_sum=0.0
x_coord3_count=0

mean_xCoord3=0

for x_point3 in xCoord3:
    x_coord3_sum = x_coord3_sum + float(x_point3)
    x_coord3_count = x_coord3_count + 1

mean_xCoord3 = x_coord3_sum/x_coord3_count

print('\nMean of X coordinates of class 3 of LS class is: ', mean_xCoord3)

y_coord3_sum=0.0
y_coord3_count=0

mean_yCoord3=0

for y_point3 in yCoord3:
    y_coord3_sum = y_coord3_sum + float(y_point3)
    y_coord3_count = y_coord3_count + 1

mean_yCoord3 = y_coord3_sum/y_coord3_count

print('\nMean of Y coordinates of class 3 of LS class is: ', mean_yCoord3)


#class1
plt.scatter(xCoord1, yCoord1, color='g', s=0.75, label='class 1')
plt.scatter(mean_xCoord1, mean_yCoord1, color='black', s=4.5, label='mean vector class1')
#class2
plt.scatter(xCoord2, yCoord2, color='r', s=0.75, label ='class 2')
plt.scatter(mean_xCoord2, mean_yCoord2, color='black', s=4.5, label='mean vector class2')
#class3
plt.scatter(xCoord3, yCoord3, color='b', s=0.75, label='class 3')
plt.scatter(mean_xCoord3, mean_yCoord3, color='black', s=4.5, label='mean vector class3')

        
plt.xticks(rotation=75)
plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.legend()
        
plt.show()

        

