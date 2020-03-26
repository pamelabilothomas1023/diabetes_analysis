import pickle
import csv
import itertools
import sys
import statistics

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import count


def ReinforcedDAGBuilder(patientIDs, df, comp):
    """
   Builds a DAG whose nodes are diagnosis codes and (directed) edges are the trajectories of diagnoses across visits.
    Each edge is updated with the number of patients who conform with that particular trajectory.
    All noise is taken care of by adding a wildcard node at each depth.

    INPUT:
        allvisits: dict of pandas dataframes. {patient_id:[df_of_diagnoses]}
        outcomesdict: dict of patient outcomes. {patient_id: 1 or 0}. 1 means heart failure, 0 means no heart failure (...yet)
        noduplicates: if True, don't consider repeat diagnosis codes.
        oldestfirst: if True, the DAG is directed in a chronological order. For a retrospective model (latest first), set this to False.
        verbose: word vomit :) Prepare your output to be overloaded. Great if you're only running a small model.
    OUTPUT:
        edgelist: dict of edges. {(from_node,to_node): {edge attrs} }
    """

    # edgelist is empty
    edgelist = {}
    df = df.toPandas()
    for patient_num, patient_id in enumerate(patientIDs):
        if (patient_num % 100 == 0):
            # for counting purposes
            print(patient_num, " out of ", len(patientIDs), " ", comp)
        # filter out the specific patient
        patient_visits = df[df['STUDYID'] == patient_id]
        num_visits = patient_visits.shape[0]
        for depth in range(num_visits):
            # for each combination of k and depth
            for k in range(depth + 1, num_visits):
                if (k != depth):
                    # take the most recent visit...
                    sid, _visit_i, _visit_i_time, d = patient_visits.iloc[depth, :].values.tolist()
                    # and the visit right before that one...
                    sid, _visit_j, _visit_j_time, d = patient_visits.iloc[(k), :].values.tolist()
                    # make edges between each of the diagnoses. The wildcards are there to make sure you account for noisy diagnoses
                    # having diagnoses made at the same time creates cycles
                    if (_visit_i != _visit_j):
                        # create edge
                        if (_visit_i_time <= _visit_j_time):
                            edgelist = createedge(_visit_i, _visit_j, _visit_i_time, _visit_j_time, edgelist, d)
                        else:
                            edgelist = createedge(_visit_j, _visit_i, _visit_j_time, _visit_i_time, edgelist, d)
    return edgelist


def createedge(_visit_i, _visit_j, _visit_i_time, _visit_j_time, edgelist, _outcome):
    # creates an edge between the visits
    # data simplification
    # create nodes which is the visit diagnosis concatenated with the time at which it was made
    _edgesp1 = str(_visit_i) + "_" + str(_visit_i_time)
    _edgesp2 = str(_visit_j) + "_" + str(_visit_j_time)
    # Create what the diagnosis is
    # Give every node a start for simplicity's sake - this way each path could start at any diagnosis
    _edges = list(itertools.product([_edgesp1], [_edgesp2]))
    for e in _edges:
        if e not in edgelist:
            # if e is not there, be sure to add one patient
            edgelist[e] = {'num_patients': 1, 'num_pos': 0, 'num_neg': 0}
        else:
            # if it is there, add to the number of patients
            edgelist[e]['num_patients'] += 1
            # add the prognosis
        if _outcome == 0:
            # they are not diagnosed
            edgelist[e]['num_neg'] += 1
        else:
            # they are diagnosed
            edgelist[e]['num_pos'] += 1
    return edgelist


if __name__ == '__main__':
    spark = SparkSession.builder.appName("clin_var").getOrCreate()
    sc = spark.sparkContext
    # Read all of the dataframes
    diagnosis = spark.read.option("header", "true").option("inferschema", "true").csv(
        str(sys.argv[1]) + "_diagnosis_dataframe_train.csv")
    clinvar = spark.read.option("header", "true").option("inferschema", "true").csv(
        str(sys.argv[1]) + "_clinvar_dataframe_train.csv")
    clinvarstatus = spark.read.option("header", "true").option("inferschema", "true").csv(
        str(sys.argv[1]) + "_clinvar_status_dataframe_train.csv")
    demographics = spark.read.option("header", "true").option("inferschema", "true").csv(
        str(sys.argv[1]) + "_demographics_dataframe_train.csv")

    diagnosis.show()
    clinvar.show()
    clinvarstatus.show()
    demographics.show()

    # union everything together
    df = diagnosis.unionAll(clinvar)
    df = df.unionAll(clinvarstatus)
    df = df.unionAll(demographics)

    # Get the patient IDs to be studied
    patientIDs = spark.read.option("header", "true").option("inferschema", "true").csv(
        str(sys.argv[1]) + "_studyids_train.csv")
    # We don't need to know their diagnosis status - that is in the other dataframes
    patientIDs = patientIDs.drop(patientIDs['Diagnosed'])
    # Create a list of the output
    patientIDs = patientIDs.rdd.flatMap(lambda x: x).collect()
    # Filter the dataframe such that it only contains patients in the training dataset
    df = df.where(df['STUDYID'].isin(patientIDs))
    # Group the number of patients and how  many diagnoses they have
    dfgroups = df.groupby(df['STUDYID'], df['Diagnosed']).agg(count(df['CODE']))
    # Filter out those that were not diagnosed
    dfnotdiagnosed = dfgroups.where(df['Diagnosed'] == 0)
    # Collect the number of counts for each code
    medcount = dfnotdiagnosed.select(dfnotdiagnosed['count(CODE)']).collect()
    # Transform into a readable list
    medcount = [i[0] for i in medcount]
    # Get the median value
    med = statistics.median(medcount)
    # Output the upper limit for diagnoses - in our case, 2*median - 5 (to create a range on either side of the median)
    med = med + (med - 5)
    # Write the median to file
    upperlimit = open(str(sys.argv[1]) + '_upperlimit.txt', 'w')
    upperlimit.write(str(med))
    upperlimit.close()
    # Filter out those that have the number of diagnoses/lab results OVER the median
    dfgroups = dfgroups.where(dfgroups['count(CODE)'] > med)
    # Filter back to the dataframe
    df = df.join(dfgroups, on=['STUDYID'], how='leftanti')
    df = df.sort('CODE')
    # Write which visits ended with patients developing the diagnosis
    beforegloriousdag = ReinforcedDAGBuilder(patientIDs, df, sys.argv[1])
    # Create a csv file for further analysis
    beforecsvfile = open(str(sys.argv[1]) + '_output.csv', 'w')
    for e in beforegloriousdag:
        # Write the output to a csv file
        beforecsvfile.write(str(e[0]) + "," + str(beforegloriousdag[e]['num_patients']) + "," + str(
            beforegloriousdag[e]['num_neg']) + "," + str(beforegloriousdag[e]['num_pos']) + "," + str(e[1]) + "\n")

    # Dump the DAG
    fbefore = open(str(sys.argv[1]) + '_output.p', 'wb')
    pickle.dump(beforegloriousdag, fbefore)
    fbefore.close()

    beforecsvfile.close()
    spark.stop()
