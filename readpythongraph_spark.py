import pickle
import networkx as nx
import sys
import math

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import collect_list, concat_ws, sum


def func(graph):
    # load data
    graphname = str(graph) + "_output.p"
    # Read all of the dataframes
    diagnosis = spark.read.option("header", "true").option("inferschema", "true").csv(
        graph + "_diagnosis_dataframe_train.csv")
    clinvar = spark.read.option("header", "true").option("inferschema", "true").csv(
        graph + "_clinvar_dataframe_train.csv")
    clinvarstatus = spark.read.option("header", "true").option("inferschema", "true").csv(
        graph + "_clinvar_status_dataframe_train.csv")
    demographics = spark.read.option("header", "true").option("inferschema", "true").csv(
        graph + "_demographics_dataframe_train.csv")

    # Union everything together
    visits_df = diagnosis.unionAll(clinvar)
    visits_df = visits_df.unionAll(clinvarstatus)
    visits_df = visits_df.unionAll(demographics)

    # Get the patient IDs and their diagnosis
    patientIDswithdiagnosis = visits_df.select(visits_df['STUDYID'], visits_df['Diagnosed']).distinct()
    # Get only the patient IDs
    patientIDs = patientIDswithdiagnosis.select(patientIDswithdiagnosis['STUDYID'])
    # Turn the patient IDs into a list
    patientIDs = patientIDs.rdd.flatMap(lambda x: x).collect()

    limit = len(patientIDs) * 0.01
    # Get the visits
    visits_df = visits_df.where(visits_df['STUDYID'].isin(patientIDs))
    # Open the DAG
    gloriousdag = pickle.load(open(graphname, "rb"))
    # Create the days index column
    visits_df = visits_df.select(concat_ws("_", visits_df['CODE'], visits_df['DAYS_INDEX']).alias("CODE_DAYS"),
                                 visits_df['STUDYID'], visits_df['Diagnosed'])
    # Drop duplicates
    visits_df = visits_df.dropDuplicates()

    # Create fraction minority
    gloriousdag_ci = {e: {'frac_minority': (gloriousdag[e]['num_pos'] / gloriousdag[e]['num_patients']),
                          'num_patients': gloriousdag[e]['num_patients']} for e in gloriousdag}
    # Get the number of patients diagnosed
    kiddiagnosed = patientIDswithdiagnosis.where(patientIDswithdiagnosis['Diagnosed'] == 1)
    # Get the total number of patients
    numpatients = len(patientIDs)
    # Get the number of patients diagnosed
    numkiddiagnosed = kiddiagnosed.select(kiddiagnosed['STUDYID']).count()
    # Get the number of patients not diagnosed
    numkidnotdiagnosed = numpatients - numkiddiagnosed
    # Create input format into graph
    gloriousedges = []
    # Create a csv file for the output
    csvfile = open(graph + "pruned_graph.csv", "w")
    # Change the visits to a Pandas dataframe so it is easier
    visits_df = visits_df.toPandas()
    # This creates the number of positive and negative patients for each diagnosis after cleaning.  I have to do this afterwards because I need a stable number of nodes
    minsupport = limit
    print("min support:", minsupport)
    sizedict = {}
    zscoredict = {}
    for e in gloriousdag_ci:
        # Get the number of patients diagnosed
        diagnosed = gloriousdag_ci[e]["frac_minority"] * gloriousdag_ci[e]["num_patients"]
        # Get the number of patients not diagnosed
        notdiagnosed = gloriousdag_ci[e]["num_patients"] - diagnosed
        # Create the confidence interval
        phat = (diagnosed + notdiagnosed) / (numkiddiagnosed + numkidnotdiagnosed)
        phatone = diagnosed / numkiddiagnosed
        phattwo = notdiagnosed / numkidnotdiagnosed
        # Get the z score
        try:
            z = (phatone - phattwo) / math.sqrt(phat * (1 - phat) * ((1 / numkiddiagnosed) + (1 / numkidnotdiagnosed)))
        except:
            z = 0
        if (abs(z) > 1.96):
            if e[0] not in sizedict:
                e0_df = visits_df[(visits_df['CODE_DAYS']) == e[0]]
                e0_df_size = e0_df.shape[0]
                sizedict[e[0]] = {'size': e0_df_size}
                e0numberpositive = e0_df['Diagnosed'].sum()
                # Take the difference and put them in node_attrs_clean
                e0numbernegative = e0_df_size - e0numberpositive
                e0phat = (e0numberpositive + e0numbernegative) / (numkiddiagnosed + numkidnotdiagnosed)
                e0phatone = e0numberpositive / numkiddiagnosed
                e0phattwo = e0numbernegative / numkidnotdiagnosed
                e0z = (e0phatone - e0phattwo) / math.sqrt(
                    e0phat * (1 - e0phat) * ((1 / numkiddiagnosed) + (1 / numkidnotdiagnosed)))
                zscoredict[e[0]] = {'z': e0z}
            else:
                e0_df_size = sizedict[e[0]]['size']
                e0z = zscoredict[e[0]]['z']
            if e[1] not in sizedict:
                e1_df = visits_df[(visits_df['CODE_DAYS']) == e[1]]
                e1_df_size = e1_df.shape[0]
                sizedict[e[1]] = {'size': e1_df_size}
                e1numberpositive = e1_df['Diagnosed'].sum()
                # Take the difference and put them in node_attrs_clean
                e1numbernegative = e1_df_size - e1numberpositive
                e1phat = (e1numberpositive + e1numbernegative) / (numkiddiagnosed + numkidnotdiagnosed)
                e1phatone = e1numberpositive / numkiddiagnosed
                e1phattwo = e1numbernegative / numkidnotdiagnosed
                e1z = (e1phatone - e1phattwo) / math.sqrt(
                    e1phat * (1 - e1phat) * ((1 / numkiddiagnosed) + (1 / numkidnotdiagnosed)))
                zscoredict[e[1]] = {'z': e1z}
            else:
                e1_df_size = sizedict[e[1]]['size']
                e1z = zscoredict[e[1]]['z']
            if ((e1_df_size > minsupport) & (e0_df_size > minsupport)):
                gloriousedges.append(
                    str(e[0]) + " " + str(e[1]) + " " + str(gloriousdag_ci[e]["frac_minority"]) + " " + str(
                        gloriousdag_ci[e]["num_patients"]) + " " + str(z))
                csvfile.write(
                    str(e[0]) + "," + str(zscoredict[e[0]]) + "," + str(gloriousdag_ci[e]["frac_minority"]) + "," + str(
                        gloriousdag_ci[e]["num_patients"]) + "," + str(e[1]) + "," + str(zscoredict[e[1]]) + "," + str(
                        z) + "\n")
    csvfile.close()

    # Create graph and output longest paths
    G = nx.parse_edgelist(gloriousedges, nodetype=str, create_using=nx.DiGraph(),
                          data=(('frac_minority', float), ('num_patients', int), ('z', float)))

    # Start out with nodes that have not been manipulated
    node_attrs_clean = {n: {"Diagnosis": 0, "No Diagnosis": 0} for n in G.nodes()}

    i = 0
    # Get the nodes from the graph
    nodes = G.nodes()
    # Create an empty list because it won't cycle through nodes
    dictlist = []
    for key in nodes:
        dictlist.append(key)
    # This creates the number of positive and negative patients for each diagnosis after cleaning.  I have to do this afterwards because I need a stable number of nodes
    minsupport = limit
    for n in dictlist:
        # Insert each node into graph
        if ((i % 10) == 0):
            print(i, " out of ", len(dictlist))
        n_df = visits_df[(visits_df['CODE_DAYS']) == n]
        # Get the number of patients
        numberofpatients = n_df.shape[0]
        # number of those with a positive diagnosis acheived by summing the Diagnosed column
        numberpositive = n_df['Diagnosed'].sum()
        # Take the difference and put them in node_attrs_clean
        numbernegative = numberofpatients - numberpositive
        # Insert the number of positive nodes
        node_attrs_clean[n]["Diagnosis"] = numberpositive
        # Insert the number of negative nodes
        node_attrs_clean[n]["No Diagnosis"] = numbernegative
        node_attrs_clean[n]["z"] = zscoredict[n]['z']
        i = i + 1

    # Set the nodes for class distribution
    nx.set_node_attributes(G, node_attrs_clean, 'class_distribution')

    # Create the information gain graph
    infogaintitle = graph + "InfoGain.p"
    cleannodestitle = graph + "nodes.p"
    InfoGainG = open(infogaintitle, 'wb')
    pickle.dump(G, InfoGainG)
    InfoGainG.close()

    # Create the clean nodes graph
    cleannodes = open(cleannodestitle, "wb")
    pickle.dump(node_attrs_clean, cleannodes)
    cleannodes.close()


if __name__ == '__main__':
    spark = SparkSession.builder.appName("clin_var").getOrCreate()
    sc = spark.sparkContext
    func(sys.argv[1])
    spark.stop()
