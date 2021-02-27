# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:28:31 2018

@author: fhp7
"""
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle

# Define plotting function
def plot_gpd_on_osm(lane_df, edge_df, node_df, title, font_size=30):
    """Plots a geodataframe of bike lanes on top of geodataframes of OSM 
    nodes and edges."""
    matplotlib.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(1)
    plt.title(title)
    node_df.plot(alpha=0.5, color='green', ax=ax)
    edge_df.plot(alpha=0.5, color='red', ax=ax)
    lane_df.plot(color='blue', ax=ax)
    plt.xlabel("Longitude (UTM Meters)")
    plt.ylabel("Latitude (UTM Meters)")
    plt.show()
    
def plot_snapped_linestrings(original, result):
    """Plot shapely linestrings against each other to check for snapping."""
    fig, axis = plt.subplots(1)
    for line in original:
        x,y = line.xy
        plt.plot(x, y, 'o', color='green')
        plt.plot(x, y, color='green')
    x,y = result.xy
    plt.plot(x, y, 'o', color='red')
    plt.plot(x, y, color='red')
    plt.show()
    
def pickle_op(op, name, item=None, folder="C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model"):
    """ Load or dump pickle files into or out of the scripts of CEE 4620 Final Project
    
    Args:
        op: string denoting the operation to perform: "load" or "dump"
        name: the python name of the object to be loaded or dumped
        item: the python object to be dumped
        folder: string denoting the absolute path of the file folder which the
                    object should be loaded from or dumped to.
    
    Returns:
        obj: only if op == "load", otherwise returns nothing
    """
    filename = name + ".p"
    filepath = os.path.join(folder, filename)
    if op == "load":
        obj = pickle.load(open(filepath, 'rb'))
        return obj
    elif op == "dump":
        pickle.dump(item, open(filepath, 'wb'))