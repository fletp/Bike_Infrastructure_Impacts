# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:13:05 2018

@author: fhp7
"""
#%%###############################################################################
### Section 1: Import data from OpenStreetMap and Motivate International, Inc
################################################################################
# Import packages
import osmnx
import pickle
import os
import pandas as pd
import numpy as np
import shapely as shp
import geopandas as gpd
import networkx as nx
from patsy import dmatrices
import statsmodels.api as sm

# Import modules for this project
import output as out
import laneprocess as lproc
from analyzeprocess import find_lane_frac, find_adj_val


#%% Download network from OpenStreetMap, then plot to verify
print("Loading OSM Network")
bike_network = osmnx.graph_from_place('New York, New York, USA', network_type = 'bike')
fig, ax = osmnx.plot_graph(bike_network, fig_height=6, node_size=2, node_alpha=0.5,
                        edge_linewidth=0.3, save=True, dpi=100, filename='manhattan_bike_network_image.png')

bike_network = osmnx.project_graph(bike_network)
# Save networkx version of the file as a pickled file
pickle.dump(bike_network, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/OSM_Network.p", 'wb'))

#%% Import CitiBike trips from CSV files. These will give the stations for
###     the year as well as the raw data for the regression.
print("Reading CSV files for CitiBike Trip Data")
citibike_file_list = os.listdir("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/NYC_Trip_Data")

# Retrieveing all files
citibike_stations_df = pd.DataFrame(columns=['stationid', 'longitude', 'latitude', 'month_installed'])
citibike_monthly_bitrips_df = pd.DataFrame()
for i in range(0, len(citibike_file_list)):
    print("Processing month number ", str(i), " of 60")
    # Read the next month's CSV file into a pandas dataframe
    print("    Read CSV file to dataframe")
    filepath = os.path.join("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/NYC_Trip_Data", citibike_file_list[i])
    new_month_df = pd.read_csv(filepath)
    
    # Rename the columns of the monthly dataframe
    print("    Rename columns")
    old_columns = new_month_df.columns
    new_columns = []
    for name in old_columns:
        name = name.lower()
        name = name.replace(" ", "")
        new_columns.append(name)
    new_month_df.columns = new_columns
    
    # Using the fact that CitiBike monthly CSV data is grouped into months
    #   by starttime, so all the records in a given CSV will have the same
    #   month for their starttime, assign all the records in the new_month_df
    #   the same month, the month of the first starttime in the dataframe. The
    #   month is stored in the format yyyymm as an integer
    month_datetime = pd.to_datetime(new_month_df.loc[0, 'starttime'])
    month_int = month_datetime.year*100 + month_datetime.month
    
    ### Find unique stations with respective geographic coordinates
    print("    Record new stations")
    # Rename and concatenate stations into a list that is pairing-agnostic
    start_station_df = new_month_df[['startstationid', 'startstationlatitude', 'startstationlongitude']]
    start_station_df = start_station_df.rename(columns={'startstationid':'stationid',
                                                        'startstationlatitude':'latitude',
                                                        'startstationlongitude':'longitude'})
    end_station_df = new_month_df[['endstationid', 'endstationlatitude', 'endstationlongitude']]
    end_station_df = end_station_df.rename(columns={'endstationid':'stationid',
                                                    'endstationlatitude':'latitude',
                                                    'endstationlongitude':'longitude'})
    new_stations_df = pd.concat([start_station_df, end_station_df])
    
    # Find all unique stations for the time period of interest
    new_stations_df = new_stations_df.loc[~new_stations_df.duplicated()]
    new_stations_df = new_stations_df.reset_index(drop=True)
    
    # Find whether these stations are new this month
    test_stations_df = new_stations_df.merge(citibike_stations_df[['stationid', 'longitude', 'latitude']], 
                                                 on=['stationid', 'longitude', 'latitude'],
                                                 how='left',
                                                 indicator=True)
    add_stations_df = new_stations_df.loc[test_stations_df['_merge'] == 'left_only']
    add_stations_df['month_installed'] = month_int
    
    # Add the new unique stations to the main stations dataframe
    citibike_stations_df = pd.concat([citibike_stations_df, add_stations_df])
    
    ### Summarize trips by station-pair-month
    print("    Group into station-pair-months")
    new_month_df['month'] = month_int
    
    # Groupby start station, end station, and month
    monthly_trips_ser = new_month_df.groupby(['startstationid', 'endstationid', 'month'])['month'].count()
    monthly_trips_ser = monthly_trips_ser.rename('count')
    monthly_trips_df = monthly_trips_ser.reset_index()
    
    # Add together trips for OD pairs so that they count both A->B and B->A movements
    monthly_trips_df['stationidA'] = monthly_trips_df.apply(lambda x: min(x.startstationid, x.endstationid), axis=1)
    monthly_trips_df['stationidB'] = monthly_trips_df.apply(lambda x: max(x.startstationid, x.endstationid), axis=1)
    monthly_bitrips_ser = monthly_trips_df.groupby(['stationidA', 'stationidB', 'month'])['count'].sum()
    monthly_bitrips_ser = monthly_bitrips_ser.rename('count')
    monthly_bitrips_df = monthly_bitrips_ser.reset_index()
    monthly_bitrips_df['month'] = monthly_bitrips_df['month'].astype('int')
    
    # Add these station-pair-month observations to the main station-pair-month dataframe
    citibike_monthly_bitrips_df = pd.concat([citibike_monthly_bitrips_df, monthly_bitrips_df])

citibike_stations_df = citibike_stations_df.reset_index(drop=True)
citibike_stations_df = citibike_stations_df[['stationid', 'longitude', 'latitude', 'month_installed']]
citibike_monthly_bitrips_df = citibike_monthly_bitrips_df.reset_index(drop=True)

#%% Write the constructed dataframes to serialized files
pickle.dump(citibike_monthly_bitrips_df, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/citibike_monthly_bitrips_df.p", 'wb'))
pickle.dump(citibike_stations_df, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/citibike_stations_df.p", 'wb'))


#%%###############################################################################
### Section 2: Prepare the "bike-conscious" network and reference dataframes
################################################################################
print("BEGINNING NETWORK PREPARATION")
# Load the OpenStreetMap network in NetworkX format and convert to GeoDataFrame format
bike_network = pickle.load(open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/OSM_Network.p", 'rb'))

# Convert to GeoDataFrame and clean up columns
osm_nodes_df, osm_edges_df = osmnx.graph_to_gdfs(bike_network)
osm_edges_df = osm_edges_df[['u', 'v', 'name', 'oneway', 'highway', 'access',
                             'bridge', 'tunnel', 'junction', 'lanes', 'service',
                             'width', 'est_width', 'key', 'maxspeed', 'ref',
                             'length', 'geometry', 'osmid']]
osm_nodes_df = osm_nodes_df[['osmid', 'highway', 'ref','x', 'y', 'geometry']]

# Merge all OSM nodes into a single multi-point to facilitate snapping
multinode_obj = osm_nodes_df.geometry.unary_union

#%% Read in New York City Bike Network Shapes file, originally from the NYC Open Data Portal
lanes_df = gpd.read_file('C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/NYC_Bike_Infrastructure_Shapefiles/geo_export_18e6e3c4-3321-4062-929b-49cd655d0a1b.shp')
lanes_df = lanes_df[lanes_df.boro != 5] # No bikesharing stations are on Staten Island
lanes_df = osmnx.project_gdf(lanes_df)

# Re-order columns and drop installation and modification times, which are all midnight anyway
lanes_df.loc[:,'instdate'] = pd.to_datetime(lanes_df.loc[:,'date_instd'], infer_datetime_format = True)
lanes_df.loc[:,'moddate'] = pd.to_datetime(lanes_df.loc[:,'date_modda'], infer_datetime_format = True)
lanes_df = lanes_df[['street', 'fromstreet', 'tostreet', 'boro', 'instdate',
                     'moddate', 'onoffst', 'allclasses', 'bikedir', 'lanecount',
                    'ft_facilit', 'tf_facilit', 'comments', 'geometry', 'segment_id']]


#%% Clean up NYC bike infrastructure data in three steps:
###     1) Join short segments together into longer bike lanes (which all share
###         the same attributes) using GeoPandas dissolve function.
###     2) Snap the vertices of the segments to the OSM nodes. This makes the
###         short segments into lines that touch end to end and allows for
###         geographic joining of OSM and NYC Cycling data later.
###     3) Merge the shorter segments (which now touch end-to-end), together into
###         longer segments using Shapely LineMerge.

## Merge attribute-sharing NYC Cycling Infrastructure lines with shapely.ops.linemerge
common_lane_attributes = ['street', 'fromstreet', 'tostreet', 'boro', 'instdate', 
                          'moddate', 'onoffst', 'allclasses', 'bikedir', 'lanecount']
#TODO: later enhancement could be to re-attach these fields. As of 5/1/18, they will be dropped.
sparse_lane_attributes = ['ft_facilit', 'tf_facilit', 'comments']

# Perform merge then clean up resulting GeoDataFrame
lanes_merged_df = lanes_df.dissolve(by= common_lane_attributes)
lanes_merged_df.reset_index(inplace = True)
lanes_merged_df = lanes_merged_df.drop(['segment_id']+sparse_lane_attributes, axis = 1)


#%% Perform snapping of NYC Cycling infrastructure: this takes several minutes
print("Snapping Cycling Infrastructure to OSM Intersections")
## Snap and linemerge all NYC Cycling lanes
lanes_snapped_df = lanes_merged_df.copy()
for i in range(0, len(lanes_merged_df)):
    original = lanes_merged_df.loc[i, 'geometry']
    snapped = shp.ops.snap(original, multinode_obj, 10)
    if isinstance(snapped, shp.geometry.linestring.LineString):
        result = snapped
    elif isinstance(snapped, shp.geometry.multilinestring.MultiLineString):
        result = shp.ops.linemerge(snapped)
    else:
        print("Alert: Bad object in snapping step!")
    lanes_snapped_df.at[i, 'geometry'] = result

pickle.dump(lanes_snapped_df, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/lanes_snapped_df.p", 'wb'))

#%% Tag each OSM edge with the NYC Cycling Lanes that it has running along it.
###     See the lane_processing.py file for much more detail on how this is achieved.
bike_edges_df = lproc.osm_bike_edge_tag(lanes_snapped_df, osm_nodes_df, osm_edges_df)

# Merge the two dataframes based on the NYC Cycling index now stored in osm_bike_edges_df
lanes_join_df = lanes_snapped_df[['instdate', 'moddate', 'onoffst', 'allclasses', 'bikedir', 'lanecount']]
final_edges_complete_df = bike_edges_df.merge(lanes_join_df, how='left', left_on='lane_id', right_index=True)

pickle.dump(final_edges_complete_df, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/final_edges_complete_df.p", 'wb'))

#%% Checking that the process worked successfully
# Plotting networks with NYC Cycling infrastructure highlighted
out.plot_gpd_on_osm(lanes_df, osm_edges_df, osm_nodes_df, "Lanes Before Processing")

disp_df = final_edges_complete_df.loc[pd.notnull(final_edges_complete_df.lane_id)]
out.plot_gpd_on_osm(disp_df, osm_edges_df, osm_nodes_df, "Fully Processed Lanes")

# Some lanes definitely were lost, and here they are to facilitate future improvements
#TODO: check if this has to do with the minimum snapping distance
lane_ids_retained = final_edges_complete_df.lane_id.unique().astype('float')
lane_ids_retained = lane_ids_retained[~np.isnan(lane_ids_retained)].astype('int')

lost_lanes_df = lanes_snapped_df.loc[~lanes_snapped_df.index.isin(lane_ids_retained), :]

out.plot_gpd_on_osm(lost_lanes_df, osm_edges_df, osm_nodes_df, "Lost Lanes")


#%% Add CitiBike Station data in the following steps:
###     1) Snap the vertices of the segments to the OSM nodes. This makes the
###         short segments into lines that touch end to end and allows for
###         geographic joining of OSM and NYC Cycling data later.
###     2) Use an intersect spatial join to attach the station properties
###         to the respective OSM nodes.

# Load CitiBike trip data
citibike_stations_df = pickle.load(open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/citibike_stations_df.p", 'rb'))

# Convert to GeoDataFrame
citibike_stations_df['geometry'] = citibike_stations_df.apply(lambda row: shp.geometry.Point(row.longitude, row.latitude), axis=1)
citibike_stations_df = gpd.GeoDataFrame(citibike_stations_df)
citibike_stations_df.crs = {'init' :'epsg:4326'} # Set the CRS to the WGS84 standard
citibike_stations_df = citibike_stations_df.to_crs(osm_edges_df.crs)

#%% Snap stations to their nearest OSM intersections
print("Snapping CitiBike Stations to OSM Intersections")
def iter_snap(new_geom, old_geom, snap_tolerance):
    if isinstance(new_geom, shp.geometry.point.Point):
        return new_geom
    else:
        snapped = shp.ops.snap(old_geom, multinode_obj, snap_tolerance)
        if snapped.equals(old_geom):
            return new_geom
        else:
            return snapped

snapped_stations_df = citibike_stations_df.copy()
snapped_stations_df.loc[:,'new_geometry'] = gpd.GeoSeries(crs=snapped_stations_df.crs)
snap_perf_list = []
for tol in range(5, 105, 5):
    print("Tolerance: ", str(tol))
    snapped_stations_df.loc[:,'new_geometry'] = snapped_stations_df.apply(lambda row: iter_snap(row.new_geometry, row.geometry, tol), axis=1)
    snap_perf_list.append(len(snapped_stations_df.loc[snapped_stations_df.new_geometry.notnull()]))
print(snap_perf_list)

# Reset the geometry column to be the snapped geometries
snapped_stations_df = snapped_stations_df.loc[snapped_stations_df['new_geometry'].notnull()]
snapped_stations_df = snapped_stations_df.set_geometry('new_geometry', drop=True)
pickle.dump(snapped_stations_df, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/snapped_stations_df.p", 'wb'))

# Join the CitiBike station attributes onto the OSM network nodes
bike_nodes_df = gpd.sjoin(osm_nodes_df, snapped_stations_df, how='left', op='intersects')


#%% Rebuild new NetworkX graph from adjusted node and edge GeoDataFrames
bike_network = lproc.gdfs_to_graph(bike_nodes_df, final_edges_complete_df, "New York City Bicycle Network, UTM Projection")
pickle.dump(bike_network, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/bike_network.p", 'wb'))

#%% Generate a reference for station-OSM node matching
stations_ref_df = bike_nodes_df.loc[bike_nodes_df['stationid'].notnull()]
stations_ref_df = stations_ref_df[['stationid', 'osmid', 'month_installed', 'longitude', 'latitude']]

# Clean up stations_ref_df by giving correct types
stations_ref_df['stationid'] = stations_ref_df['stationid'].astype('int')
stations_ref_df['osmid'] = stations_ref_df['osmid'].astype('int64')

# Remove duplicates, keeping the earliest instances of multiply snapped stations
stations_ref_df = stations_ref_df.sort_values(by=['month_installed'])
stations_ref_df = stations_ref_df.loc[~stations_ref_df.duplicated(['stationid', 'osmid'])]
# Capture this moment for debugging purposes
pre_mult_snap_df = stations_ref_df.copy()
# Continue with the code
stations_ref_df = stations_ref_df.loc[~stations_ref_df.duplicated(['stationid'])]

# Set index and save stations_ref_df
stations_ref_df = stations_ref_df.set_index(['stationid'], drop=False)
pickle.dump(stations_ref_df, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/stations_ref_df.p", 'wb'))

#%% Verify that the snap and join worked
# Find stations which were lost in the join, dropping stations at coordinates
#   (0, 0) because we don't know how to assign them anyway.
lost_station_selector = stations_ref_df['stationid'].values.tolist()
lost_stations_df = snapped_stations_df.loc[~snapped_stations_df['stationid'].isin(lost_station_selector)]
lost_stations_df = lost_stations_df[(lost_stations_df.latitude != 0) & \
                                    (lost_stations_df.longitude != 0)]

# Check for stations which were snapped to multiple OSM nodes, this is because
#   they were moved over time. To deal with this, I assume that the earlier
#   station coordinates are the ones used throughout this study. Incorrect,
#   but unavoidable, and only applies to 30 stations or so.
mult_snap_df = pre_mult_snap_df.loc[pre_mult_snap_df.duplicated(['stationid'], keep=False)]

# Check for OSM nodes with multiple stations snapped to them, this might
#   not be a problem if the earlier stations are discontinued before
#   the later stations are installed, but I can't see that. In any case,
#   there are only about 30 of them.
mult_station_df = stations_ref_df.loc[stations_ref_df.duplicated(['osmid'], keep=False)]

# Unique OSMs
num_OSMs = stations_ref_df['osmid'].unique()

#%% Make figures to show results of verification
# Conclusion from figure. Lost stations are in New Jersey, on Colonel's Row Island, or outside of NYC proper. 
out.plot_gpd_on_osm(lost_stations_df, osm_edges_df, osm_nodes_df, "Lost Stations")

# For multiply snapped stations
mult_snap_df['geometry'] = mult_snap_df.apply(lambda row: shp.geometry.Point(row.longitude, row.latitude), axis=1)
mult_snap_df = gpd.GeoDataFrame(mult_snap_df)
mult_snap_df.crs = {'init' :'epsg:4326'} # Set the CRS to the WGS84 standard
mult_snap_df = mult_snap_df.to_crs(osm_edges_df.crs)
out.plot_gpd_on_osm(mult_snap_df, osm_edges_df, osm_nodes_df, "Multiply Snapped")


#%%###############################################################################
### Section 3: Perform analysis using "bike-conscious" network and CitiBike trip data
################################################################################
print("BEGINNING DATA ANALYSIS")
# Load the bike network prepared previously
bike_network = pickle.load(open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/bike_network.p", 'rb'))
stations_ref_df = pickle.load(open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/stations_ref_df.p", 'rb'))
citibike_monthly_bitrips_df = pickle.load(open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/citibike_monthly_bitrips_df.p", 'rb'))


#%% Group by unique pair of station ids (not OSMids)
###     in order to reduce the number of paths and other columns
###     computed multiple times. Then apply inclusion criteria to further limit
###     the required computation, especially of shortest paths
total_biods_ser = citibike_monthly_bitrips_df.groupby(['stationidA', 'stationidB'])['count'].sum()
total_biods_ser = total_biods_ser.rename('count')
total_biods_df = total_biods_ser.reset_index()

# Add OSM ids to the total_biods_df
total_biods_df['stationosmidA'] = total_biods_df['stationidA'].map(stations_ref_df['osmid'])
total_biods_df['stationosmidB'] = total_biods_df['stationidB'].map(stations_ref_df['osmid'])


### Apply inclusion criteria
incl_biods_df = total_biods_df.copy()
print("Initial:\n", incl_biods_df['count'].describe(), "\n")
# Take away all stations that have no associated OSM id. These stations
#   probably occur in New Jersey or on some of the island parks. See Section 2.
#   for more details. No paths can be computed between these.
incl_biods_df = incl_biods_df.loc[pd.notnull(incl_biods_df.stationosmidA) & \
                                  pd.notnull(incl_biods_df.stationosmidB)]
print("Stations must have been matched to OSM:\n", incl_biods_df['count'].describe(), "\n")


# Take away OD pairs where the origin and destination are the same,
#   because there will be no path for these
incl_biods_df = incl_biods_df.loc[incl_biods_df.stationosmidA != \
                                  incl_biods_df.stationosmidB]
print("No self-paths:\n", incl_biods_df['count'].describe(), "\n")

# Take away OD pairs with less than a certain threshold of trips
#   between them in the study period
activity_threshold = 500
incl_biods_df = incl_biods_df[incl_biods_df['count'] > activity_threshold]
print("With activity threshold:\n", incl_biods_df['count'].describe(), "\n")

#TODO: I wish I had been able to apply the infrastructure change criterion first
#           before applying the number of riders criteria, but unfortunately
#           it wasn't possible with the time and computing resources I had.

#%% Find shortest paths between each pair of stations with more than activity_threshold trips annually
print("Finding shortest paths")
incl_biods_df['path'] = incl_biods_df.apply(lambda x: nx.dijkstra_path(bike_network, 
                                                                       source=x.stationosmidA,
                                                                       target=x.stationosmidB,
                                                                       weight='length'), axis=1)
# Save the incl_biods_df
pickle.dump(incl_biods_df, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/incl_biods_df.p", 'wb'))


#%% Merge all calculated columns in the incl_biods_df onto the monthly_bitrips_df
###     This also has the effect of applying the exclusion criteria found
###     for total_biods_df above, because only idtuples found in both dataframes
###     will be preserved.
monthly_final_df = pd.merge(citibike_monthly_bitrips_df, 
                            incl_biods_df[['stationidA', 'stationidB', 'path']],
                            how='inner',
                            on=['stationidA', 'stationidB'])
# Set a multiindex for monthly_final_df
monthly_final_df = monthly_final_df.set_index(keys=['stationidA', 'stationidB', 'month'], drop=False)

#%% Traverse shortest paths and calculate percentage (then add to monthly_final_df)
print("Calculating Lane Fractions")
monthly_final_df['lanefrac'] = monthly_final_df.apply(lambda x: find_lane_frac(path=x.path, 
                                                                                month=x.month,
                                                                                network=bike_network), axis=1)
#TODO: debug which edges got the lane information, because this is a multidigraph with coordinates [o, d, k]

#%% Add monthly total trips and shortest paths to monthly_final_df
# Find total activity in each month
monthly_total_activity_ser = citibike_monthly_bitrips_df.groupby(['month'])['month'].count()
monthly_total_activity_ser = monthly_total_activity_ser.rename('count')

# Add monthly total trips to the monthly_final_df
monthly_final_df['totaltrips'] = monthly_final_df['month'].map(monthly_total_activity_ser)


#%% Perform forward finite difference calculations for each month on several columns
# Check that all index combinations return a unique row
num_duplicates = monthly_final_df.index.duplicated().sum()
if num_duplicates > 0:
    print("Warning! Duplicate indices in monthly_final_df!")
# Construct a set of all the unique months in monthly_final_df
valid_months = set(monthly_final_df['month'].unique())

### Find associated value for nearby months
print("Computing forward finite differences")
query_step = 1 # Find values for the following month

# Assumes no change in lane fraction if no data is available
monthly_final_df['nextlanefrac'] = monthly_final_df.apply(lambda x: find_adj_val(df=monthly_final_df,
                                                                                    stationidA=x.stationidA,
                                                                                    stationidB=x.stationidB,
                                                                                    month=x.month,
                                                                                    colname='lanefrac',
                                                                                    step=query_step,
                                                                                    default_value=x.lanefrac,
                                                                                    valid_months=valid_months), 
                                                                                axis=1)
monthly_final_df['nextcount'] = monthly_final_df.apply(lambda x: find_adj_val(df=monthly_final_df,
                                                                                    stationidA=x.stationidA,
                                                                                    stationidB=x.stationidB,
                                                                                    month=x.month,
                                                                                    colname='count',
                                                                                    step=query_step,
                                                                                    default_value=0,
                                                                                    valid_months=valid_months), 
                                                                                axis=1)
monthly_final_df['nexttotaltrips'] = monthly_final_df.apply(lambda x: find_adj_val(df=monthly_final_df,
                                                                                    stationidA=x.stationidA,
                                                                                    stationidB=x.stationidB,
                                                                                    month=x.month,
                                                                                    colname='totaltrips',
                                                                                    step=query_step,
                                                                                    default_value=np.nan,
                                                                                    valid_months=valid_months), 
                                                                                axis=1)
# Take differences between following month and the current month
monthly_final_df['difflanefrac'] = monthly_final_df['nextlanefrac'].subtract(monthly_final_df['lanefrac'])
monthly_final_df['diffcount'] = monthly_final_df['nextcount'].subtract(monthly_final_df['count'])
monthly_final_df['difftotaltrips'] = monthly_final_df['nexttotaltrips'].subtract(monthly_final_df['totaltrips'])

pickle.dump(monthly_final_df, open("C:/Users/fhp7/Desktop/Cornell/CEE 4620/Final Project/Model/monthly_final_df.p", 'wb'))


#%% Perform regression on difference data
print("Performing Regression")
# Remove all records that have no infrastructure change
regress_df = monthly_final_df.loc[(monthly_final_df.difflanefrac != 0) & \
                                  (pd.notnull(monthly_final_df.difflanefrac))]

regress_df['logdifflanefrac'] = regress_df['difflanefrac'].apply(np.log)

# Use patsy to generate design matrix and target vector
y, X = dmatrices('diffcount ~ difflanefrac + difftotaltrips', data=regress_df, return_type = 'dataframe')

# Fit the model using statsmodels and print the results
result = sm.OLS(y, X).fit()
print(result.summary())