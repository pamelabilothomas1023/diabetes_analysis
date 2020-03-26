import pickle
import networkx as nx
import operator
from scipy.stats import entropy
import sys

from pyspark.sql import SparkSession
from pyspark import SparkContext
import pandas as pd


def lookup(item, df):
    df = df[(df["Converted"] == str(item))]
    return df.iloc[0, 1]


def InfoGain(distr1, distr2):
    ig = -1.0
    try:
        ig = entropy(distr1) - entropy(distr2)
    except:
        pass
    return ig


def func(graph):
    InfoGainName = graph + "InfoGain.p"
    NodeName = graph + "nodes.p"
    G = pickle.load(open(InfoGainName, "rb"))
    node_attrs_clean = pickle.load(open(NodeName, "rb"))

    dia = pd.read_csv("icdcodes.csv")

    dia["Converted"] = dia["Converted"].apply(lambda x: str(x)[0:3])

    ### WARNING: INFO GAIN IS OFFSET BY A FACTOR OF LOG_e(2) TO AVOID NEGATIVE VALUES

    # Here I am creating the information gain information for regular infogain, asymmetric infogain, and offset infogain using nx.set_edge_attributes


    edge_infogain_clean = {e: 0 for e in G.edges()}
    for u, v in edge_infogain_clean:
        edge_infogain_clean[(u, v)] = InfoGain([node_attrs_clean[u]["No Diagnosis"], node_attrs_clean[u]["Diagnosis"]],
                                               [node_attrs_clean[v]["No Diagnosis"], node_attrs_clean[v]["Diagnosis"]])
    nx.set_edge_attributes(G, edge_infogain_clean, 'infogain')


    # class-agnostic
    edge_infogains_sorted = sorted(edge_infogain_clean.items(), key=operator.itemgetter(1), reverse=True)

    # Print the highest information gain edges

    edge_infogains_sorted = edge_infogains_sorted[0:20]

    print("Highest information gains:")

    for e, ig in edge_infogains_sorted:
        try:
            if ("DIA" in e[0]):
                utext = lookup(e[0][0:3], dia) + " " + str(e[0])[len(e[0]) - 1] + " years"
            else:
                utext = e[0] + " years"
            if ("DIA" in e[1]):
                vtext = lookup(e[1][0:3], dia) + " " + str(e[1])[len(e[1]) - 1] + " years"
            else:
                vtext = e[1] + " years"
            print(utext, node_attrs_clean[e[0]], "\n\t", "information gain: ", ig, "\n", vtext, node_attrs_clean[e[1]])
        except:
            print(e[0], node_attrs_clean[e[0]], "\n\t", "information gain: ", ig, "\n", e[1], node_attrs_clean[e[1]])
        print()

    print("****************")

    GetTopConfidenceNodes(G, dia)

    print("****************")

    GetTopConfidenceEdges(G, dia)


def PrintCodeDescr(g, dia, codes, mode="edge"):
    """Prints verbal descriptions for nodes and edges of ICD codes"""
    if mode == "edge":
        print("Highest confidence edges:")
        for e in codes:
            (u, v), y, w, x = e
            try:
                if ("DIA" in u):
                    utext = lookup(u[0:3], dia) + " " + str(u)[len(u) - 1]
                else:
                    utext = u
                if ("DIA" in v):
                    vtext = lookup(v[0:3], dia) + " " + str(v)[len(v) - 1]
                else:
                    vtext = v
                print(utext + " years " + str("====>") + vtext + " years ",
                      ("\t") + (str(x)) + ' percent ' + str(w) + ' patients ' + str(y) + ' confidence')
                print("\t", nx.get_node_attributes(g, "class_distribution")[u], "====>",
                      nx.get_node_attributes(g, "class_distribution")[v], "\n")
            except:
                print(u + " years " + str("====>") + v + " years ",
                      ("\t") + (str(x)) + ' percent ' + str(w) + ' patients ' + str(y) + ' confidence')
                print("\t", nx.get_node_attributes(g, "class_distribution")[u], "====>",
                      nx.get_node_attributes(g, "class_distribution")[v], "\n")
    elif mode == "node":
        print("Highest confidence nodes:")
        for n in codes:
            u, w = n
            if ("DIA" in u):
                utext = lookup(u[0:3], dia) + " " + str(u)[len(u) - 1]
            else:
                utext = u
            try:
                print(utext + " years " + "\t" + str(w))
            except:
                print(utext + " years " + "\t" + str(w))
                pass
            print("\t", nx.get_node_attributes(g, "class_distribution")[u], "patients\n")


def GetTopConfidenceNodes(g, dia, topn=20):
    """Computes confidence for all nodes, then finds top n confidence-of-heart-failure nodes"""
    conf_hf = {}
    for i in g.nodes(data=True):
        n, distr = i
        if (distr['class_distribution']['No Diagnosis'] != 0):
            conf_hf[n] = distr['class_distribution']['Diagnosis'] / (
                        distr['class_distribution']['Diagnosis'] + distr['class_distribution']['No Diagnosis'])
    nx.set_node_attributes(g, conf_hf, 'confidence')
    nodeconf = {i[0]: i[1]['confidence'] for i in g.nodes(data=True) if
                (i[1]['class_distribution']['No Diagnosis'] != 0)}
    nodeconf_sorted = sorted(nodeconf.items(), key=operator.itemgetter(1), reverse=True)
    PrintCodeDescr(g, dia, nodeconf_sorted[:topn], mode="node")


def GetTopConfidenceEdges(g, dia, topn=20):
    """Finds top n confidence-of-heart-failure edges"""
    edgez = {(e[0], e[1]): e[2]['z'] for e in g.edges(data=True)}
    edgeconf = {(e[0], e[1]): e[2]['frac_minority'] for e in g.edges(data=True)}
    edgenum = {(e[0], e[1]): e[2]['num_patients'] for e in g.edges(data=True)}
    edgez_sorted = sorted(edgez.items(), key=operator.itemgetter(1), reverse=True)[:topn]
    newedgez_sorted = []
    for e in edgez_sorted:
        e = list(e)
        edge = e[0]
        e.append(edgenum[e[0]])
        e.append(edgeconf[e[0]])
        newedgez_sorted.append(e)
    PrintCodeDescr(g, dia, newedgez_sorted, mode="edge")


if __name__ == '__main__':
    spark = SparkSession.builder.appName("clin_var").getOrCreate()
    sc = spark.sparkContext
    func(sys.argv[1])
    spark.stop()
