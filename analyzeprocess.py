# -*- coding: utf-8 -*-
"""
Created on Sun May  6 22:39:51 2018

@author: fhp7
"""
import numpy as np

# for each OD pair of nodes:
#   while not yet at the end of the path
#       find length of edge between cur_node and prev_node
#       add the edge length to the total edge length counter
#       if this edge has a bike lane
#           add the edge length to the bike lane length counter
#   record the bike_lane_length/total_path_length in the citibike_monthly_od_ser next to the normalized trip count
def find_lane_frac(path, month, network):
    """Find the percentage of bike lane by length along the shortest path
        between two osmids.
        
    Args:
        path: list containing the nodes on the shortest path between start and
                end nodes
        month: integer representation of the month in yyyymm format
    Returns:
        frac: float fraction of the length of this shortest path covered by
                any sort of bike lane.
    """
    path_list = path.copy()
    cur_node = path_list.pop(0)
    if len(path_list) == 0:
        return np.nan
    else:
        total_len = 0
        lane_len = 0
        while len(path_list) > 0:
            prev_node = cur_node
            cur_node = path_list.pop(0)
            cur_edge_dict = network.edges[prev_node, cur_node, 0]
            total_len = total_len + cur_edge_dict['length']

            if 'instdate' in cur_edge_dict:
                #TODO: replace this with a permanent fix by adding a month column
                #   to nxprep.py
                month_int = cur_edge_dict['instdate'].year*100 + cur_edge_dict['instdate'].month
                if month_int <= month:
                    lane_len = lane_len + cur_edge_dict['length']
        frac = lane_len/total_len
        return frac
    
def find_adj_val(df, stationidA, stationidB, month, colname, step, default_value, valid_months):
    """
    Args:
        df: pandas dataframe holding columns for month, startstationosmid,
                and endstationosmid
        stationidA: CitiBike station id of the A-station for this station pair
        stationidB: CitiBike station id of the B-station for this station pair
        month: integer representing the month of the year from 1 to 12
        colname: string denoting the column in which the previous value is to
                    be found
        step: integer representing the number of months to look forward into
                the future or back into the past. Negative values look into
                the past.
        default_value: the value that should be returned if the query_month is
                        valid but no data exists for it. This situation will
                        occur when no trips were taken on a given od pair
                        during a valid month.
        valid_months: set of all the unique months represented in df
    Returns:
        val: the value of the column given in colname for the station ids and 
                months specified. If that value does not exist in df, then
                apply a default value or a nan value
    """
    query_month = month + step
    try:
        val = df.loc[(stationidA, stationidB, query_month), colname]
    except KeyError: # That is, if that month didn't exist in the index
        if month in valid_months: # If the month was valid but there were no trips
            val = default_value
        else: # If the month was invalid
            val = np.nan
    
    return val
    
def find_prepost_val(df, colname, month, step, idt, default_value):
    """
    Args:
        df: pandas dataframe holding columns for month, startstationosmid,
                and endstationosmid
        colname: string denoting the column in which the previous value is to
                    be found
        month: integer representing the month of the year from 1 to 12
        step: integer representing the number of months to look forward into
                the future or back into the past. Negative values look into
                the past.
        idt: tuple denoting the ids (not OSM) of the end stations for this route
        default_value: the value that should be returned if the query_month is
                        valid but no data exists for it. This situation will
                        occur when no trips were taken on a given od pair
                        during a valid month.
    Returns:
        val: float fraction of the shortest path between the given nodes that
                    was covered by bike lanes in the previous month.
    """
    valid_months = [1,2,3,4,5,6,7,8,9,10,11,12]
    query_month = month + step
    
    if query_month in valid_months:
        val_ser = df.loc[(df['idtuple'] == idt) & \
                         (df['month'] == query_month), colname]
        if len(val_ser) > 1:
            print("Warning! Multiple possible data points found in val_ser!")
            print(val_ser)
        elif len(val_ser) == 1:
            val = val_ser.iloc[0]
        else: # That is, if there were no trips on an od pair in a valid month
            val = default_value
    else:
        val = np.nan
    
    return val