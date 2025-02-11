# -*- coding: utf-8 -*-
"""
Analysis of biomechanical gait patterns in elderly subjects using a 
robotic walker for hip fracture rehabilitation.

This script takes an .xlsx that stores the information from a ROM signal
and calculates the gait parameters from it.

@author: Alfonso Rafael Gordon Cabello de los Cobos
"""

# Imported libraries
import glob

import pandas as pd
import os

import numpy as np

from scipy.signal import find_peaks

import matplotlib.pyplot as plt

"""
This function checks the data in the column from a patient, which is 
the ROM from one of the legs, and return an array storing the index at which 
maximuns and minimuns where detected.
"""
def peak_detection(leg, legId, key):
    #First we find all the peaks of the function
    maxPeaks, _ = find_peaks(leg, height=np.mean(leg), distance=10)
    
    #Since the find_peaks function can't detect the peaks coresponding to the
    #minumuns
    #   the columns must be wsiped for finding them
    columnaGirada = leg * (-1)
    minPeaks, _ = find_peaks(columnaGirada, height=np.mean(columnaGirada),distance=10)
    
    #The array will store the times of maximuns and minimuns in order
    peakOrderSequence = [] 
    actualMax = 0
    actualMin = 0
    actualCol = 0    

    
    for actualCol in np.arange(len(leg)):
        if (len(peakOrderSequence) <= (len(maxPeaks) + len(minPeaks))):

            if(actualCol == maxPeaks[actualMax]):
                peakOrderSequence.append(actualCol)
                if(actualMax < len(maxPeaks) - 1):
                    actualMax += 1 

            if(actualCol == minPeaks[actualMin]):    
                peakOrderSequence.append(actualCol)
                if(actualMin < len(minPeaks) - 1):
                    actualMin += 1
    
    #Some peaks migth be errors while getting the ROM signal
    #   that are detected by find_peaks so they must be filtered by getting
    #   only the peaks that are above the average
    media = np.mean(leg)
    upperBound = media + 2
    lowerBound = media - 2
    realMax = False 
    realPeakSequence = []
  #     saco el primero
    for picos in np.arange(len(peakOrderSequence)):
        if (leg[peakOrderSequence[picos]] > upperBound or leg[peakOrderSequence[picos]] < lowerBound):
            if(leg[peakOrderSequence[picos]] > media):
               realMax = True         
            cummulatedPeaks = []
            cummulatedPeaks.append(peakOrderSequence[picos])
            
            for picosPosteriores in np.arange(picos + 1, len(peakOrderSequence)):
                if((leg[peakOrderSequence[picosPosteriores]] > media) == realMax):
                    cummulatedPeaks.append(peakOrderSequence[picosPosteriores])
                else: break
            
            if(realMax == True):
                realPeakSequence.append(max(cummulatedPeaks))
            else: realPeakSequence.append(min(cummulatedPeaks))
            realMax = False

    x_axis = np.arange(len(leg))
    x_axis1 = np.arange(len(realPeakSequence))
    plt.figure()
    plt.title("Real ROM " + legId + key)
    plt.plot(x_axis, leg)
    y = [media] * len(x_axis)
    plt.plot(x_axis, y)

    plt.figure()
    plt.title("Simplified ROM " + legId + key)
    plt.plot(x_axis1, leg[realPeakSequence])
    y = [media] * len(x_axis1)
    plt.plot(x_axis1, y)
    
    return realPeakSequence
    


"""
This function gets all the ROM values from a patient stored in the dictionary

"""
def getDataPatient(key):
    leftLeg = dictionaryRom.get(key).iloc[:,3].values
    rightLeg = dictionaryRom.get(key).iloc[:,4].values
    times = dictionaryRom.get(key).iloc[:,2].values
    return leftLeg, rightLeg, times



"""
This function calculates the number of steps a patient had performed.
Since the functions SecuenciaPicos and segundaSecuenciaPicos store in an array 
at which moment are the Max and Min values of the hip ROM we can calculate 
the number of steps just by dividing the length of the final array by 2 since 
each pair of Max and Min corresponds to an step.

"""
def calculateSteps(gaitSimplifiedSequenceLeft, gaitSimplifiedSequenceRight):
    stepsL = len(gaitSimplifiedSequenceLeft) / 2
    stepsR = len(gaitSimplifiedSequenceRight) / 2
    totalSteps = stepsL + stepsR
    
    return totalSteps, stepsL, stepsR


"""
This function calculates the average cadence, number of steps made by minute, 
by getting the number of steps and dividing it by the time of the session

"""
def calculateCadence(steps, times):
    cadence = steps / ((times[-1] - times[0]) / 1000) #[steps/s]
    finalCadence = cadence * 60 #The time is recorded in seconds and the candence
                                # is usually calculated in steps/minute.
    return finalCadence #[steps/min]
    
"""
This function calculates the average step length by dividing the 
distance walked, which is asked to the therapist, by the number of steps 
made by the patient.
Also the average stride length is calculated by the sum of 3 steps since is 
the distance between two consecutive steps of the same foot.

"""
def calculateStepAndStrideLength(steps):
    while True:
        distance_txt = input("Insert the walked distance in meters:\n") #[m]
    
        if distance_txt.isdigit() and int(distance_txt) > 0:
            distance = int(distance_txt)
            break   
    averageStepLength = distance / steps
    averageStrideLength = averageStepLength * 2
    return averageStepLength, averageStrideLength, distance

"""
This function calculates the times between each step. Since the data stored in
gaitSimplifiedSequence are the time of peak of the stance phase followed by 
the peak of the swing phase, the we must only take into acount the stance peaks
for calculating the time.

"""
def calculateStepTime(gaitSimplifiedSequenceLeft, gaitSimplifiedSequenceRight, time):
    timesL = []
    timesR = []
    for i in range(len(gaitSimplifiedSequenceLeft) - 2):
        if i < len(gaitSimplifiedSequenceLeft) - 2 and i % 2 == 0:
            timesL.append((time[gaitSimplifiedSequenceLeft[i + 1]] - time[gaitSimplifiedSequenceLeft[i]]) / 1000) #ms to s
            
    for i in range(len(gaitSimplifiedSequenceRight) - 2):
        if i < len(gaitSimplifiedSequenceRight) - 2 and i % 2 == 0:
            timesR.append((time[gaitSimplifiedSequenceRight[i + 1]] - time[gaitSimplifiedSequenceRight[i]]) / 1000)
    
    averageTimesL = sum(timesL)/len(timesL)
    averageTimesR = sum(timesR)/len(timesR)
    
    return averageTimesL, averageTimesR

"""
This function calculates the single and double support times using 
the patient's whole data. First we calculate the total degrees by summing the 
absolute values from the left and right legs. Then we get the indexes at wich 
this sum is less or is higher or equal to 10º for classifying into two different
list. The stance time contains a full Single support and two double support periods

"""
def calculateSingleStanceAndDoubleSupportTimes(leftLeg, rightLeg, times):
    sst = []
    dst = []
    averageSst = averageDst = 0

    legDifference = abs(leftLeg) + abs(rightLeg)
    
    indexListDst = [i for i in range(len(legDifference)) if legDifference[i] < 10]
    indexListSst = [i for i in range(len(legDifference)) if legDifference[i] >= 10]

    start = None
    end = None

    for i in range(len(indexListDst)-1):
        if indexListDst[i]+1 == indexListDst[i+1]:
            if start is None:
                start = i
            end = i+1
        elif start is not None:
            dst.append((times[indexListDst[end]] - times[indexListDst[start]]) / 1000)
            start = None
            end = None
   
    start = None #sometimes the variables dont reset after finishing with the calculation of the DST
    end = None
    
    for j in range(len(indexListSst)-1):
        if indexListSst[j]+1 == indexListSst[j+1]:
            if start is None:
                start = j
            end = j+1
        elif start is not None:
            sst.append((times[indexListSst[end]] - times[indexListSst[start]]) / 1000)
            start = None
            end = None
      
    averageSst = sum(sst)/len(sst)
    averageDst = sum(dst)/len(dst)
    averageStance = averageDst * 2 + averageSst
    

    return averageSst, averageDst, averageStance

"""
This function calculates the stride time.

"""
def calculateStrideTime(gaitSimplifiedSequenceLeft, gaitSimplifiedSequenceRight, leftLeg, rightLeg, times):
    averageSTL = averageSTR = 0
    strideTimesLeft = []
    strideTimesRight = []
    avgRomL = sum(leftLeg) / len(leftLeg)
    avgRomR = sum(rightLeg) / len(rightLeg)
    
    for i in range(len(gaitSimplifiedSequenceLeft) - 1):

        if leftLeg[gaitSimplifiedSequenceLeft[i]] < avgRomL:
            strideTimesLeft.append((times[int(gaitSimplifiedSequenceLeft[i])] - times[0]) / 1000)
        
    for i in range(len(gaitSimplifiedSequenceRight)- 1):
        if rightLeg[gaitSimplifiedSequenceRight[i]] < avgRomR:
            strideTimesRight.append((times[int(gaitSimplifiedSequenceRight[i])]  - times[0]) / 1000)
    
    differenceStrideTimesLeft = []
    differenceStrideTimesRight = []
    
    for j in range(len(strideTimesLeft) - 2):
        differenceStrideTimesLeft.append(strideTimesLeft[j + 1] - strideTimesLeft[j])
    
    for j in range(len(strideTimesRight) - 2):
        differenceStrideTimesRight.append(strideTimesRight[j + 1] - strideTimesRight[j])

    averageSTL = sum(differenceStrideTimesLeft) / len(differenceStrideTimesLeft)
    averageSTR = sum(differenceStrideTimesRight) / len(differenceStrideTimesRight)
    totalAvgST = (averageSTL + averageSTR)/2
    
    return averageSTL, averageSTR, totalAvgST

"""
This function calculates the stride and gait speed. It takes the Stride time, 
length and distance, which are already calculated in previous functions, and 
changes the units to cm before performing the calculations.

"""
def calculateGaitStrideSpeed(stride_time, stride_length, distance):
    newDistance = distance * 100 #[m] to [cm]
    newSL = stride_length * 100
    gaitSpeed = newDistance / stride_time
    strideSpeed = newSL / stride_time
    return gaitSpeed, strideSpeed


"""
This function calculates the swing time. It can be calculated by dividing the 
step length by the speed since time = space/speed

"""
def calculateSwingTime(stepLength, gaitSpeed):
    swingTime = stepLength / gaitSpeed
    return swingTime


#---------------------------------------------------------------------------------------------------

# Preparing the data access
# Asking for the directory for extracting the ROM signals
path = input("Where are the ROM signals stored?\n")

# Takes every xlsx on the folder, each xlsx is a ROM signal

so = input("Which OS are you using?\n1. Windows\n2. Linux\n")
if so == 1:
    files = glob.glob(path + "\*.xlsx")
else:
    files = glob.glob(path + "/*.xlsx")
    
dictionaryRom = {}

data_frame = pd.DataFrame()

# Storing the data of each ROM file into the dictionary,
#   the key is only the filename and the result is the dataset.
for filename in files:
    dictionaryRom[filename[(len(path) + 1):(len(filename) - 5)]] = pd.read_excel(filename)
    #filename[(len(path) + 1):(len(filename) - 5)] isolates only the name of
    #   the file (excluding directory and file sufix)  
    #There are multiple columns per patient.  Column 3 stores the time sequence, 4 stores the Left Hip ROM,
    #   Column 5 stores Right Hip ROM and Column 6 stores the Weight

#Calculates the gait parameters for every patient on each leg
for key in dictionaryRom.keys():
    print("-----------------" + key + "-----------------\n")
    leftLeg, rightLeg, times = getDataPatient(key)
    simplifiedLeftLeg = peak_detection(leftLeg, "left ", key)
    simplifiedRightLeg = peak_detection(rightLeg, "right ", key)
    
    plt.figure()
    plt.title("Legs ROM combined")
    plt.plot(leftLeg, label = "Left leg")
    plt.plot(rightLeg, label = "Right leg")
    plt.legend()
    plt.xlim(100, 201)
    
    totalSteps, stepsL, stepsR = calculateSteps(simplifiedLeftLeg, simplifiedRightLeg)
    print("Total steps: " + str(totalSteps) + "\n" + "Total left leg steps: " + str(stepsL) + "\n" + "Total right leg steps: " + str(stepsR) + "\n" )
    
    cadence = calculateCadence(totalSteps, times)
    print("Cadence [steps/min]: " + str(cadence) + "\n")

    averageStepLength, averageStrideLength, distance = calculateStepAndStrideLength(totalSteps)
    print("Average Step Length [cm]: " + str(averageStepLength) + "\n" + "Average Stride Length [cm]: " + str(averageStrideLength) + "\n" + "Distance [m]: " + str(distance) + "\n" )

    timesL, timesR = calculateStepTime(simplifiedLeftLeg, simplifiedRightLeg, times)
    print("Average Step time left leg [s]: " + str(timesL) + "\n" + "Average Step time right leg [s]: " + str(timesR) + "\n")
    
    averageSst, averageDst, averageStance = calculateSingleStanceAndDoubleSupportTimes(leftLeg, rightLeg, times)
    print("Average Single Support time [s]: " + str(averageSst) + "\n" + "Average Double Support time [s]: " + str(averageDst) + "\n" + "Average Stance time [s]: " + str(averageStance) + "\n" )

    averageSTL, averageSTR, totalAvgST = calculateStrideTime(simplifiedLeftLeg, simplifiedRightLeg, leftLeg, rightLeg, times)
    print("Average Stride time left leg [s]: " + str(averageSTL) + "\n" + "Average Stride time right leg [s]: " + str(averageSTR) + "\n" + "Average Stride time [s]: " + str(totalAvgST) + "\n")

    
    gaitSpeed, strideSpeed = calculateGaitStrideSpeed(totalAvgST, averageStrideLength, distance)
    print("Gait speed [cm/s]: " + str(gaitSpeed) + "\n" + "Stride speed [cm/s]: " + str(strideSpeed) + "\n")

    swingTime = calculateSwingTime(averageStepLength, gaitSpeed)
    print("Swing time [s]: " + str(swingTime) + "\n")
    
    results = {
        'Filename': key,
        'Total Steps': totalSteps,
        'Steps - Left Leg': stepsL,
        'Steps - Right Leg': stepsR,
        'Cadence': cadence,
        'Avg Step Length': averageStepLength,
        'Avg Stride Length': averageStrideLength,
        'Distance': distance,
        'Avg Step Time - Left Leg': timesL,
        'Avg Step Time - Right Leg': timesR,
        'Avg Single Support Time': averageSst,
        'Avg Double Support Time': averageDst,
        'Avg Stance Time': averageStance,
        'Avg Stride Time - Left Leg': averageSTL,
        'Avg Stride Time - Right Leg': averageSTR,
        'Avg Stride Time': totalAvgST,
        'Avg Gait Speed': gaitSpeed,
        'Avg Stride Speed': strideSpeed,
        'Swing Time': swingTime
    }
    data_frame = data_frame.append(results, ignore_index=True)


# Create the full path to the new directory
new_directory_path = os.path.join(path, "RESULTS")

# Create the new directory if it doesn't already exist
if not os.path.exists(new_directory_path):
    os.makedirs(new_directory_path)
    
#Ask for the file name to the user and creating a CSV file with that name
file_name = input("Name of the file for storing the results: ")
result_file = file_name + '.csv'

# Save results to CSV file
csv_file = os.path.join(new_directory_path, result_file)
data_frame.to_csv(csv_file, index=False)

#Changing file tipe to XLSX
result_file = file_name + '.xlsx'

# Save results to XLSX file
xlsx_file = os.path.join(new_directory_path, result_file)
data_frame.to_excel(xlsx_file, index=False)

print("A CSV and XLSX file was generated into the directory " + new_directory_path)

"""
TODO
Remaining:
    •	Step width [cm]: Lateral distance from heel center to line of progression. Line of progression is formed as the line formed by two consecutive footprints of the opposite foot.

"""




