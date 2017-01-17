# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 21:01:49 2016

@author: Nishant.Gupta
"""
import pandas as pd

########## Read CSV File ############
def readCSVFile(reading_directory,filename):
    data=pd.read_csv(reading_directory+'/'+filename)
    return data

########## Write Datframe Into CSVFile ############
def writeCSVFile(data,writing_directory,filename):
    data.to_csv(writing_directory+'/'+filename,index=False)   

reading_directory='C:/Nishant/Personal/Assignment/Final/Scores'
writing_directory='C:/Nishant/Personal/Assignment/Final/Scores'

data=readCSVFile(reading_directory,"SampleDataTFIDF(2)Scores.csv")

for index, row in data.iterrows():
    print(index)
    line=str(row['CharacterisedText']).strip()
    
    number=str(row['CharacterisedNumber'])
    if number=='nan':
        number="0"
    else:
        number=number.strip()
    
    lowest_number=int(number.split(' ')[-1])
#    print(lowest_number)
    try:
        i=0
        for word in line.split(' '):
            if word!=' ':
                current_number=int(number.split(' ')[i])
                data.loc[index,word]=current_number/lowest_number
                i+=1
    except Exception as e:
        print("e",index)
        

writeCSVFile(data,writing_directory,"Try.csv")





