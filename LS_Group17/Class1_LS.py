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

def size(x_C):
    x_c_count = 0
    for i in x_C:
        x_c_count = x_c_count + 1
        
    return x_c_count


def meanCalc(xC,yC):
    x_coord_sum=0.0
    x_coord_count=0
    y_coord_sum = 0.0
    y_coord_count = 0

    mean_xCoord=0.0
    mean_yCoord = 0.0

    for x_point in xC:
        x_coord_sum = x_coord_sum + float(x_point)
        x_coord_count = size(xC)
        
    for y_point in yC:
        y_coord_sum = y_coord_sum + float(y_point)
        y_coord_count = size(yC)

    mean_xCoord = x_coord_sum/x_coord_count
    mean_yCoord = y_coord_sum/y_coord_count
    
    return  mean_xCoord,mean_yCoord

def cov_mat_Calc(xC,yC,dV):
    
    cov_sumXY=0.0
    cov_sumXX = 0.0
    cov_sumYY = 0.0
    
    cov_XX = 0.0
    cov_XY = 0.0
    cov_YY = 0.0
    
    result = meanCalc(xC, yC)
    mean_x, _ = result
    _, mean_y = result
    
    for x,y in dV:
        cov_sumXY = cov_sumXY + (x - mean_x) * (y - mean_y)
        cov_sumXX = cov_sumXX + (x - mean_x) * (x - mean_x)
        cov_sumYY = cov_sumYY + (y - mean_y) * (y - mean_y)
        
    cov_XX = cov_sumXX/size(dV)
    cov_XY= cov_sumXY/size(dV)
    cov_YY = cov_sumYY/size(dV)
    
    cov_mat = [[cov_XX,cov_XY],[cov_XY,cov_YY]]
    
    return cov_mat

def covariance(xC,yC,dV):

    cov_sumXX = 0.0
    cov_sumYY = 0.0
    
    cov_XX = 0.0
    
    cov_YY = 0.0
    
    result = meanCalc(xC, yC)
    mean_x, _ = result
    _, mean_y = result
    
    for x,y in dV:
        cov_sumXX = cov_sumXX + (x - mean_x) * (x - mean_x)
        cov_sumYY = cov_sumYY + (y - mean_y) * (y - mean_y)
        
    cov_mat = [[cov_XX,0],[0,cov_YY]]
    
    return cov_mat
    
    
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
path1 = "Class1_train.csv"
#class2 path
path2 = "Class2_train.csv"
#class3 path
path3 = "Class3_train.csv"



#class1 opening file with path in readable mode and giving it as an alias
with open(path1, 'r') as dataset1:
    grid1 = csv.reader(dataset1) #object of csv module reads dataset
    for tupl1 in grid1:           #iteration
        try:                                      #exception handling try and catch
            xCoord1.append(float(tupl1[0]))
            yCoord1.append(float(tupl1[1]))
        except (ValueError, IndexError):
            pass
        
        
mean_x1,mean_y1 = meanCalc(xCoord1, yCoord1) 

print('\nMean vector of class 1 of LS class is:[', mean_x1,',',mean_y1,']')

##############
datavector1 = []
for i in range(size(xCoord1)):
    datavector1.append((xCoord1[i],yCoord1[i]))
        
print("\nCovariance matrix is: - ", cov_mat_Calc(xCoord1, yCoord1, datavector1))   

######################################################################################

#class2
with open(path2, 'r') as dataset2:
    grid2 = csv.reader(dataset2)
    for tupl2 in grid2:
        try:
            xCoord2.append(float(tupl2[0]))
            yCoord2.append(float(tupl2[1]))
        except (ValueError, IndexError): 
            pass
        
mean_x2,mean_y2 = meanCalc(xCoord2, yCoord2) 

print('\nMean vector of class 2 of LS class is:[', mean_x2,',',mean_y2,']')


datavector2 = []
for i in range(size(xCoord2)):
    datavector2.append((xCoord2[i],yCoord2[i]))
        
print("\nCovariance matrix is: - ", cov_mat_Calc(xCoord2, yCoord2, datavector2))
print(np.cov(xCoord2,yCoord2))
# =============================================================================
# print('\nCovariance of class 2 of LS class is:', cov_sum2/x_coord2_count)
# =============================================================================

      

#class3        
with open(path3, 'r') as dataset3: 
    grid3 = csv.reader(dataset3)
    for tupl3 in grid3:
        try: 
            xCoord3.append(float(tupl3[0]))
            yCoord3.append(float(tupl3[1]))
        except (ValueError, IndexError):
            pass
        
        
mean_x3,mean_y3 = meanCalc(xCoord3, yCoord3) 
print('\nMean vector of class 2 of LS class is:[', mean_x3,',',mean_y3,']')

datavector3 = []
for i in range(size(xCoord3)):
    datavector3.append((xCoord3[i],yCoord3[i]))
        
print("\nCovariance matrix is: - ", cov_mat_Calc(xCoord3, yCoord3, datavector3))
print(np.cov(xCoord3,yCoord3))

#class1
plt.scatter(xCoord1, yCoord1, color='g', s=0.75, label='class 1')
plt.scatter(mean_x1, mean_y1, color='black', s=4.5)
#class2
plt.scatter(xCoord2, yCoord2, color='r', s=0.75, label ='class 2')
plt.scatter(mean_x2, mean_y2, color='black', s=4.5)
#class3
plt.scatter(xCoord3, yCoord3, color='b', s=0.75, label='class 3')
plt.scatter(mean_x3, mean_y3, color='black', s=4.5, label='mean vector c1, c2, c3')

        
plt.xticks(rotation=75)
plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.legend()
        
plt.show()

# t2 = [[t1[0][0]],[t1[0][1]]] for trNAPOSE

