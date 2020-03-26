from pyspark.sql import SparkSession
from pyspark import SparkContext
import sys
import pickle
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import numpy as np
import math
import operator


def func(graph):
    # load data
    # Read all of the dataframes
    diagnosis = spark.read.option("header", "true").option("inferschema", "true").csv(
        graph + "_diagnosis_dataframe_test.csv")
    clinvar = spark.read.option("header", "true").option("inferschema", "true").csv(
        graph + "_clinvar_dataframe_test.csv")
    clinvarstatus = spark.read.option("header", "true").option("inferschema", "true").csv(
        graph + "_clinvar_status_dataframe_test.csv")
    demographics = spark.read.option("header", "true").option("inferschema", "true").csv(
        graph + "_demographics_dataframe_test.csv")

    # Union everything together
    visits_df = clinvar.unionAll(diagnosis)
    visits_df = visits_df.unionAll(clinvarstatus)
    visits_df = visits_df.unionAll(demographics)
    # Get the patient IDs to be studied
    patientIDswithdiagnosis = spark.read.option("header", "true").option("inferschema", "true").csv(
        str(sys.argv[1]) + "_studyids_test.csv")
    patientIDswithdiagnosis = patientIDswithdiagnosis.drop(patientIDswithdiagnosis['DAYS_INDEX_DX'])
    # We don't need to know their diagnosis status - that is in the other dataframes
    patientIDs = patientIDswithdiagnosis.drop(patientIDswithdiagnosis['Diagnosed'])
    # Create a list of the output
    patients = patientIDs.rdd.flatMap(lambda x: x).collect()
    # Get the true outcome for each patient
    y_true_source = patientIDswithdiagnosis.select('Diagnosed').rdd.flatMap(lambda x: x).collect()
    trainingpatientIDs = spark.read.option("header", "true").option("inferschema", "true").csv(
        str(sys.argv[1]) + "_studyids_train.csv")
    y_training_true_source = trainingpatientIDs.select('Diagnosed').rdd.flatMap(lambda x: x).collect()
    numtrainingpatients = len(y_training_true_source)
    # Get how many patients were diagnosed
    numkiddiagnosed = np.sum(y_training_true_source)
    # Get how many patients weren't diagnosed
    numkidnotdiagnosed = numtrainingpatients - numkiddiagnosed
    # Create empty dataframe for predictions
    y_pred = []  # [0 for p in patients]
    # Create empty dataframe for outcomes
    y_true = []
    # Read in the upper limit file
    upperlimitf = open(str(sys.argv[1]) + "_upperlimit.txt", "r")
    upperlimit = upperlimitf.read()
    upperlimit = float(upperlimit)
    upperlimitf.close()
    # Import the graph
    G = pickle.load(open(graph + "InfoGain.p", "rb"))
    f = open(str(sys.argv[1]) + "predictoutput_07.txt", "w")
    # Predict each patient
    for i, p in enumerate(patients):
        if ((i % 100 == 0) and (i != 0)):
            print("\t", i, "out of: ", len(patients))
            print(roc_auc_score(y_true, y_pred))
        # Get the visits for each patient
        current_visits = visits_df.where(visits_df['STUDYID'] == p)
        ct = current_visits.count()
        if ((ct > int(sys.argv[2])) & (ct < upperlimit)):
            # current_visits.show()
            pred = PredictHF(current_visits, G, numkiddiagnosed, numkidnotdiagnosed)
            y_pred += [pred]
            f.write(str(pred) + "," + str(y_true_source[i]) + "," + str(current_visits.count()) + "\n")
            y_true += [y_true_source[i]]
    f.write(str(roc_auc_score(y_true, y_pred)))
    f.close()
    return y_pred


def GetPatientSubgraph(_df, g):
    relevantnodes = []
    _df = _df.toPandas()
    for i, r in _df.iterrows():
        nodes = r.dropna().tolist()
        relevantnodes += [str(nodes[1]) + "_" + str(nodes[2])]
    g_ = nx.Graph(g.subgraph(relevantnodes))
    return g_, relevantnodes


from functools import reduce


def PredictHF(_df, g, numkiddiagnosed, numkidnotdiagnosed):
    gtest, _nodeslist = GetPatientSubgraph(_df, g)
    connected = gtest.to_undirected()
    connected = nx.connected_component_subgraphs(connected)
    edgez = {}
    edgeconf = {}
    p_HF = 0.5
    for h in connected:
        nodez = {e[0]: abs(e[1]['class_distribution']['z']) for e in h.nodes(data=True)}
        edgeza = {(e[0], e[1]): abs(e[2]['z']) for e in h.edges(data=True)}
        nodezconf = {e[0]: ((e[1]['class_distribution']['Diagnosis'] / (
                    e[1]['class_distribution']['Diagnosis'] + e[1]['class_distribution']['No Diagnosis']))) for e in
                     h.nodes(data=True)}
        edgeconfa = {(e[0], e[1]): e[2]['frac_minority'] for e in h.edges(data=True)}
        edgez.update(edgeza)
        edgez.update(nodez)
        edgeconf.update(edgeconfa)
        edgeconf.update(nodezconf)
    edgez_sorted = sorted(edgez.items(), key=operator.itemgetter(1), reverse=True)[0:7]
    newedgez_sorted = []
    for e in edgez_sorted:
        e = list(e)
        if ((e[1] > 1.96)):
            e.append(edgeconf[e[0]])
            newedgez_sorted.append(e)
    edge_weights_HF = [e[2] for e in newedgez_sorted]
    if (len(edge_weights_HF) > 3):
        # edge_weights_HF.remove(max(edge_weights_HF))
        edge_weights_HF.remove(min(edge_weights_HF))
        # edge_weights_HF.remove(min(edge_weights_HF))
        # edge_weights_HF.remove(min(edge_weights_HF))
    edge_weights_HF = [i for i in edge_weights_HF if (i != 0) and (i != 1)]
    edge_weights_NHF = [1 - w for w in edge_weights_HF]
    if len(edge_weights_HF) > 0:
        p_HF_temp = reduce(lambda x, y: x * y, edge_weights_HF)
        p_NHF_temp = reduce(lambda x, y: x * y, edge_weights_NHF)
        try:
            p_HF = (p_HF_temp / (p_HF_temp + p_NHF_temp))  # log-normalized
        except:
            print("Divide by zero error")
            p_HF = 0.5
    return p_HF


if __name__ == '__main__':
    spark = SparkSession.builder.appName("clin_var").getOrCreate()
    sc = spark.sparkContext
    func(sys.argv[1])
    spark.stop()
