import pandas as pd
import math
from nltk.corpus import stopwords
from textblob import TextBlob as tb
from nltk.stem.porter import PorterStemmer

########## Read CSV File ############
def readCSVFile(reading_directory,filename):
    data=pd.read_csv(reading_directory+'/'+filename)
    return data

########## Write Datframe Into CSVFile ############
def writeCSVFile(data,writing_directory,filename):
    data.to_csv(writing_directory+'/'+filename,index=False)   

############ Remove Stopwords and add new column in pandas dataframe as CleanedText ##########
def removeStopWordsStemming(data):
    stop = set(stopwords.words('english'))
    for index, row in data.iterrows():
        string=""
        for i in row['Text'].lower().split(): 
            if i not in stop:
                string+=stemming(i)+' '
        data.loc[index,'CleanedText']=string
    return data  
    
############### Stemming ########################################################################
def stemming(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)

########## Will calculate how many top words to keep as per the TFIDF score ############
def calculateTopTFIDFScoreWords(length):
    if length <6:
       return length
    elif length<14:
        return length*0.5
    elif length<14:
        return length*0.4
    elif length<30:
        return length*0.3
    else:    
        return length*0.2
        
#Calculate Term Frequency for a word in similar type of category levels
def tf(word, small_blob):
    return sum(1 for blob in small_blob if word in blob.words)

def n_containing(word, large_blob):
    if word in large_dictionary.keys():
        return large_dictionary[word]
    else:        
        large_dictionary[word]=sum(1 for blob in large_blob if word in blob.words)
        return large_dictionary[word]

def idf(word, large_blob):
    return math.log(len(large_blob) / (1 + n_containing(word, large_blob)))

def tfidf(word, small_blob, large_blob):
     return tf(word, small_blob) * idf(word, large_blob)

#Main Code Starts
reading_directory='C:/Nishant/Personal/Assignment/Final/Files'
writing_directory='C:/Nishant/Personal/Assignment/Final/Files'

large_dictionary={}
small_dictionary={}

data=readCSVFile(reading_directory,"SampleData.csv")

data=removeStopWordsStemming(data)

#Done only once large_blob list is created.
large_blob=[]
for index, row in data.iterrows():
    large_blob.append(tb(str(row['CleanedText'])))


for index, row in data.iterrows():
    if(index<3000):
        continue
    print(index)
    target_blob=tb(str(row['CleanedText']))

    small_blob=[]    
    for index1, row1 in data.iterrows():
        if row1['Level1']==row['Level1']:
            small_blob.append(tb(str(row1['CleanedText']))) 
            
    scores = {word: tfidf(word, small_blob,large_blob) for word in target_blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_scorers=int(calculateTopTFIDFScoreWords(len(target_blob.words)))
    
    string=""
    number=""
    for word, score in sorted_words[:top_scorers]:
        string+=word+' '
        number+=str(int(score))+' '    
    
    data.loc[index,'CharacterisedText']=string
    data.loc[index,'CharacterisedNumber']=number

######End of For Loop#############

writeCSVFile(data,writing_directory,"SampleDataLevel1_Number.csv")



