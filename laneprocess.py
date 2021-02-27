# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:26:38 2018

@author: fhp7
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely as shp
import networkx as nx

def osm_bike_edge_tag(snap_df, nodes_df, edges_df):
    """Tag OpenStreetMap edges GeoDataFrame with the ids of cycling infrastructure elements
    
    Args:
        snap_df: GeoDataFrame of cycling infrastructure elements snapped to
                    OSM nodes
        nodes_df: GeoDataFrame of OpenStreetMap nodes
        edges_df: GeoDataFrame of OpenStreetMap edges
        
    Returns:
        bike_edges_df: GeoDataFrame of OpenStreetMap edges with cycling
                        infrastructure attributes ids from snap_df tagged on it.
                        Derived from a copy of edges_df
    """
    bike_edges_df = edges_df.copy()
    bike_edges_df['lane_id'] = np.nan 
    # For each snapped cycling infrastructure LineString (could be a MultiLineString),
    #   which are each held in a row of snap_df's 'geometry' column.
    for i in range(0, len(snap_df)):
        if i%50 == 0:
            print("The loop is currently at row: ", str(i), " out of ", str(len(snap_df)))
        line_dict = shp.geometry.mapping(snap_df.loc[i, 'geometry'])
        # If the geometry is a single linestring
        if line_dict['type'] == "LineString":
            line_nodes_check(line_dict['coordinates'], bike_edges_df, nodes_df, edges_df, i)
        # If the geometry is many linestrings grouped together in a multilinestring
        elif line_dict['type'] == "MultiLineString":
            for linestring in line_dict['coordinates']:
                line_nodes_check(linestring, bike_edges_df, nodes_df, edges_df, i)
        else:
            print("Alert: Bad object type!")
    return bike_edges_df


def line_nodes_check(line_coord_tuples, bike_df, nodes_df, edges_df, lane_id):
    prev_node = 'None'
    for vertex in line_coord_tuples:
        # Detect and find osmid of OSM node concurrent with this vertex
        node_series = nodes_df.loc[(nodes_df.x == vertex[0]) & (nodes_df.y == vertex[1])].loc[:,'osmid']
        if len(node_series) > 1: # This will probably never happen.
            print("Alert: multiple OSM nodes detected!")
        elif len(node_series) == 1:
            # Get the current node id from the node_series (it is in string format)
            cur_node = node_series.iloc[0]
            
            if prev_node != 'None': # If we have two nodes to check for an edge with
                # Check for an edge, in either direction
                edge_selector = np.logical_or(np.logical_and(edges_df.u == int(prev_node),
                                                                        edges_df.v == int(cur_node)),
                                                        np.logical_and(edges_df.u == int(cur_node),
                                                                       edges_df.v == int(prev_node)))
                if edge_selector.any(): # If any edges were found between the nodes
                    bike_df.at[edge_selector, 'lane_id'] = lane_id
            # Regardless of whether there was an edge or not, advance the previous node
            prev_node = cur_node
            cur_node = 'None'


### Using adapted version of osmnx source code. The code had a bug in it and I don't want to wait to have it fixed.
def gdfs_to_graph(gdf_nodes, gdf_edges, graph_name):
    """
    Convert node and edge GeoDataFrames into a graph
    Parameters
    ----------
    gdf_nodes : GeoDataFrame
    gdf_edges : GeoDataFrame
    graph_name: the desired name of the output graph
    Returns
    -------
    networkx multidigraph
    """

    G = nx.MultiDiGraph()
    G.graph['crs'] = gdf_nodes.crs
    G.graph['name'] = graph_name

    # add the nodes and their attributes to the graph
    G.add_nodes_from(gdf_nodes.index)
    attributes = gdf_nodes.to_dict()
    for attribute_name in gdf_nodes.columns:
        # only add this attribute to nodes which have a non-null value for it
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(G, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added
    # separately) or null
    for _, row in gdf_edges.iterrows():
        attrs = {}
        for label, value in row.iteritems():
            if (label not in ['u', 'v', 'key']) and (isinstance(value, list) or pd.notnull(value)):
                attrs[label] = value
        G.add_edge(row['u'], row['v'], key=row['key'], **attrs)

    return G