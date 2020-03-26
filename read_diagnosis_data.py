import sys
import argparse
import statistics
import numpy as np

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql.functions import when, lit, concat, collect_list, collect_set, concat_ws, min, floor, max, count

def GetFirstDate(df, _unitoftime):
    #find the minimum date
    df_grouped = df.groupby([df['STUDYID'], df["CODE"]]).agg(min(df['DAYS_INDEX']))
    #convert to first date
    df_grouped = df_grouped.withColumn("DAYS_INDEX", floor((df_grouped["min(DAYS_INDEX)"].cast(FloatType()))/_unitoftime))
    #drop the minimum days
    df_grouped = df_grouped.drop(df_grouped["min(DAYS_INDEX)"])
    #Filter out all diagnoses that occured at time 0 AND also filter out the diagnosis we are looking for
    return df_grouped

if __name__ == '__main__':
    spark = SparkSession.builder.appName("clin_var").getOrCreate()
    sc = spark.sparkContext
    #Create the list of adverse codes
    adverse_code = sys.argv[1].split()
    #Read the file of type 2 diabetes patients
    diadf = spark.read.option("header", "true").option("inferschema", "true").csv("/mnt/data0/input/t2d_subgroups/t2d_diagnosis.csv")
    #Rename columns, take the first 3 numbers of the ICD code, and append DIA to the end
    diadf = diadf.select(diadf['STUDYID'], diadf['DAYS_DX_INDEX'].alias("DAYS_INDEX"), concat(diadf['DX_CODE'].substr(0,3), lit("_DIA")).alias("CODE"))
    #Only include patients who have the diagnosis
    diadf = diadf.withColumn('CODE', when(diadf['CODE'].isin(adverse_code), sys.argv[2]).otherwise(diadf['CODE']))
    #Get the first date of each diagnosis
    diadf = GetFirstDate(diadf, 365)
    #There are some diabetes diagnoses that occur AFTER the first date.  Get rid of them
    diabetesproblems = diadf.where(((diadf['CODE'] == "250_DIA") | (diadf['CODE'] == "E11_DIA")) & (diadf['DAYS_INDEX'] > 0))
    #Take the problems out of the data
    diadf = diadf.join(diabetesproblems, on=['STUDYID'], how='leftanti')
    #Filter out diagnoses that don't help us
    diadf = diadf.where((diadf['CODE'].substr(0,1) != "V") & (diadf['CODE'] != "780_DIA") & (diadf['CODE'] != '794_DIA') & (diadf['CODE'] != '790_DIA'))
    #Standardize kidney diagnoses
    diadf = diadf.where(diadf['DAYS_INDEX'] >= 0)
    dfcount = diadf.groupby(diadf['STUDYID']).agg(count(diadf['CODE']))
    dfcount = dfcount.where(dfcount['count(CODE)'] < 5)
    diadf = diadf.join(dfcount, on=['STUDYID'], how='leftanti')
    #Filter out kidney diagnosis
    dfdaydiagnosed = diadf.where((diadf['CODE'] == sys.argv[2]) & (diadf['DAYS_INDEX'] > 0))
    #Get the year that kidney disease was diagnosed
    medlist = dfdaydiagnosed.select(dfdaydiagnosed['DAYS_INDEX']).collect()
    #Take out the first element as one does when one collects in Spark
    medlist = [i[0] for i in medlist]
    #Find the median value
    a25th = np.percentile(medlist, 25)
    a75th = np.percentile(medlist, 75)
    #Filter out those who had a diagnosis before the median
    diadf = diadf.where((diadf['DAYS_INDEX'] == 0) | (diadf['CODE'] == sys.argv[2]))
    #Get the distinct study ids
    limit = dfdaydiagnosed.select('STUDYID').distinct().collect()
    #Count how many people are enrolled in the study
    limit = len(limit)
    #Create the limit of one percent of the dataframe in the study
    limit = limit*.01
    #Group by how many people have received each diagnosis
    diadfcount = diadf.groupby(diadf['CODE']).count()
    #Filter out diagnoses that occur in less than one percent of the population
    diadfcount = diadfcount.where(diadfcount['COUNT'] > limit)
    #Only filter back on the code
    diadfcount = diadfcount.select(diadfcount['CODE'])
    #Join the codes that occur in more than one percent of the population with the general dataset
    diadf = diadf.select(diadf['STUDYID'], diadf['CODE'], diadf['DAYS_INDEX'])
    #Rename the code column for joining purposes
    diadfcount = diadfcount.select(diadfcount['CODE'].alias("Code2"))
    #Join the count dataframe with the original dataframe on an inner join so that only those diagnoses that occured frequently are in the column
    diadf = diadfcount.join(diadf, diadf['CODE'] == diadfcount['Code2'], how='inner')
    #Select columns that we want
    diadf = diadf.select(diadf['STUDYID'], diadf['CODE'], diadf['DAYS_INDEX'])
    #Get the fast progressors
    dfdaydiagnosedfirst = dfdaydiagnosed.where(dfdaydiagnosed['DAYS_INDEX'] <= a25th)
    #Get the slow progressors
    dfdiagnosedlater = dfdaydiagnosed.where(dfdaydiagnosed['DAYS_INDEX'] > a75th)
    dfdaydiagnosedfirst.show()
    dfdiagnosedlater.show()
    #Get the IDs of the slow progressors
    dfids = dfdaydiagnosedfirst.select('STUDYID', dfdaydiagnosedfirst['DAYS_INDEX'].alias('DAYS_INDEX_DX')).distinct()
    #Get the IDs of the fast progressors
    dfidsnotrelids = dfdiagnosedlater.select('STUDYID', dfdiagnosedlater['DAYS_INDEX'].alias('DAYS_INDEX_DX')).distinct()
    #Join back to original idd
    dffinalrel = diadf.join(dfids, on=['STUDYID'], how='inner')
    #dffinalrel = dffinalrel.where(dffinalrel['DAYS_INDEX'] < dffinalrel['DAYS_INDEX_DX'])
    #Take out the entries that have the diagnosis we are looking at
    dffinalrel = dffinalrel.where(dffinalrel['CODE'] != sys.argv[2])
    dffinalrel.show()
    #Create the dataframe of the slow progressors
    dffinalnotrel = diadf.join(dfidsnotrelids, on=['STUDYID'], how='inner')
    dffinalnotrel = dffinalnotrel.where(dffinalnotrel['DAYS_INDEX'] < dffinalnotrel['DAYS_INDEX_DX'])
    #Take out the current diagnoses
    dffinalnotrel = dffinalnotrel.where(dffinalnotrel['CODE'] != sys.argv[2])
    #Find out how large the dataset is
    dffinalnotrel = dffinalnotrel.drop_duplicates()
    dfids = dffinalrel.select('STUDYID', 'DAYS_INDEX_DX').distinct()
    dfidsnotrelids = dffinalnotrel.select('STUDYID', 'DAYS_INDEX_DX').distinct()
    sizeofrelevants = len(dfids.collect())
    #Count the size of the study ids
    sizeofidsnotrel = len(dfidsnotrelids.collect())
    #Sample such that the datasets are the same size
    if (sizeofidsnotrel > sizeofrelevants):
        dfidsnotrelids = dfidsnotrelids.sample(False, float(sizeofrelevants)/float(sizeofidsnotrel))
    else:
        dfids = dfids.sample(False, float(sizeofidsnotrel)/float(sizeofrelevants))
    print(sizeofrelevants)
    print(sizeofidsnotrel)
    #Indicate that the value in not relevants are not fast progressors
    dffinalnotrel = dffinalnotrel.withColumn('Diagnosed', lit(0))
    #Union with the relevant dataframe
    dfids = dfids.withColumn('Diagnosed', lit(1))
    #Create the not diagnosed column for the not relevants
    dfidsnotrelids = dfidsnotrelids.withColumn("Diagnosed", lit(0))
    #Create a diagnosed column for the relevants
    dffinalrel = dffinalrel.withColumn("Diagnosed", lit(1))
    #Union the ids together
    studyids = dfids.union(dfidsnotrelids)
    #Union the relevants and not relevants together
    diadf = dffinalnotrel.union(dffinalrel)
    #Divide into test and training
    studyidstrain = studyids.sample(False, .8)
    #Get the studyids that are not in the trianing dataset
    studyidstest = studyids.join(studyidstrain, on=['STUDYID'], how='leftanti')
    #Only output the studyid and whether or not the patient is diagnosed
    studyidstest = studyidstest.select(studyidstest['STUDYID'], studyidstest['DAYS_INDEX_DX'], studyidstest['Diagnosed'])
    studyidstrain = studyidstrain.select(studyidstrain['STUDYID'], studyidstrain['DAYS_INDEX_DX'], studyidstrain['Diagnosed'])
    #Write studyids
    studyidstrain.write.format("csv").mode("overwrite").option("header", "true").save(str(sys.argv[2]) + '_studyids_train.csv', sep=',')
    studyidstest.write.format("csv").mode("overwrite").option("header", "true").save(str(sys.argv[2]) + '_studyids_test.csv', sep=',')
    #Now we only want the studyid
    studyidstest = studyidstest.select(studyidstest['STUDYID'])
    studyidstrain = studyidstrain.select(studyidstrain['STUDYID'])
    #Join the training study IDs with the rest of the dataframe
    diadftrain = diadf.join(studyidstrain, on=['STUDYID'], how='inner')
    #Select columns
    diadftrain = diadftrain.select(diadftrain['STUDYID'], diadftrain['CODE'], diadftrain['DAYS_INDEX'], diadftrain['Diagnosed'])
    #Join the testing study IDs with the rest of the dataframe
    diadftest = diadf.join(studyidstest, on=['STUDYID'], how='inner')
    #Select columns
    diadftest = diadftest.select(diadftest['STUDYID'], diadftest['CODE'], diadftest['DAYS_INDEX'], diadftest['Diagnosed'])
    #Write diagnoses to file
    diadftrain.write.format("csv").mode("overwrite").option("header", "true").save(str(sys.argv[2]) + '_diagnosis_dataframe_train.csv', sep=',')
    diadftest.write.format("csv").mode("overwrite").option("header", "true").save(str(sys.argv[2]) + '_diagnosis_dataframe_test.csv', sep=',')
    #Write the limit
    limitfile = open(str(sys.argv[2]) + "limitfile.txt", "w")
    limitfile.write(str(limit))
    limitfile.close()
    #Write the median
    medfile = open(str(sys.argv[2]) + "medfile.txt", "w")
    medfile.write(str(medlist))
    medfile.close()
    spark.stop()
