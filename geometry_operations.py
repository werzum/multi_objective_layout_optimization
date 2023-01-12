from copy import deepcopy
from turtle import distance
from shapely.geometry import LineString, Point

import matplotlib.pyplot as plt

import numpy as np
import math
import itertools

import geometry_utilities

def generate_possible_lines(road_points, target_trees, anchor_trees, overall_trees, slope_line, height_gdf, plot_possible_lines):
    """Compute which lines can be made from road_points to anchor_trees without having an angle greater than max_main_line_slope_deviation
    First, we generate all possible lines between  each point along the road and all head anchors.
    For those which do not deviate more than max_main_line_slope_deviation degrees from the slope line, we compute head anchor support trees along the lines.
    If those are present, we compute triples of tail anchor support trees.
    If those are present, valid configurations are appended to the respective lists.


    Args:
        road_points (_type_): _description_
        target_trees (_type_): _description_
        anchor_trees (_type_): _description_
        slope_line (_type_): _description_
        max_main_line_slope_deviation (_type_): How much the central of three lines can deviate from the slope
        max_anchor_distance (_type_): How far away should the anchors be at most

    Returns:
        _type_: _description_
    """    
    possible_lines = []
    slope_deviation = []
    possible_anchor_triples = []
    possible_support_trees = []
    angle_between_supports = []
    nr_of_supports = []
    location_of_int_supports = []

    max_main_line_slope_deviation = 45

    if plot_possible_lines:
        fig = plt.figure(figsize=(15, 15))
        ax = plt.axes(projection='3d') 
        plt.axis([-60,60,-100,20])
    else:
        ax = None
    
    for point in road_points:
        for target in target_trees.geometry:
            possible_line = LineString([point, target])
            possible_line_to_slope_angle = geometry_utilities.angle_between(possible_line, slope_line)

            if possible_line_to_slope_angle<max_main_line_slope_deviation:
                
                # compute the intermediate support trees betweeen the point and the target
                support_tree_candidates = generate_support_trees(overall_trees, target, point, possible_line)

                # continue if there is at least one support tree candidate
                if len(support_tree_candidates) > 0:
                    # generate a list of line-triples that are within correct angles to the road point and line candidate
                    triple_angle = generate_triple_angle(point, possible_line, anchor_trees)

                    # if we have one or more valid anchor configurations, append this configuration to the line_gdf
                    if triple_angle and len(triple_angle)>0:
                        
                        # add the corresponding values to the array
                        angle_between_supports.append(compute_angle_between_supports(possible_line, height_gdf))
                        possible_lines.append(possible_line)
                        slope_deviation.append(possible_line_to_slope_angle)
                        possible_anchor_triples.append(triple_angle)
                        possible_support_trees.append([support_tree_candidates.geometry])
                        # and finally check for height obstructions and add supports if necessary
                        returned_location_int_supports = []
                        returned_number_supports, returned_location_int_supports = no_height_obstructions(possible_line, height_gdf, 0, plot_possible_lines, ax, returned_location_int_supports)
                        nr_of_supports.append(returned_number_supports)
                        location_of_int_supports.append(returned_location_int_supports)
                        
    start_point_dict = {}
    for id,line in enumerate(possible_lines):
        start_point_dict[id] = line.coords[0]
    
    if plot_possible_lines:
        # height_gdf_small = height_gdf.iloc[::500, :]
        # ax.plot_trisurf(height_gdf_small["x"], height_gdf_small["y"], height_gdf_small["elev"], linewidth=0)
        fig.show()

    return possible_lines, slope_deviation, possible_anchor_triples, possible_support_trees, angle_between_supports, start_point_dict, nr_of_supports, location_of_int_supports

def compute_angle_between_supports(possible_line, height_gdf):
    """ Compute the angle between the start and end support of a cable road.

    Args:
        possible_line (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_:  the angle between two points in degrees
    """    
    start_point_xy, end_point_xy = Point(possible_line.coords[0]), Point(possible_line.coords[1])
    max_deviation = 0.1
    start_point_xy_height = fetch_point_elevation(start_point_xy,height_gdf,max_deviation)
    end_point_xy_height = fetch_point_elevation(end_point_xy,height_gdf,max_deviation)

    # piece together the triple from the xy coordinates and the z (height)
    start_point_xyz = (start_point_xy.coords[0][0],start_point_xy.coords[0][1],start_point_xy_height)
    end_point_xyz = (end_point_xy.coords[0][0],end_point_xy.coords[0][1],end_point_xy_height)

    # and finally compute the angle
    return geometry_utilities.angle_between_3d(start_point_xyz, end_point_xyz)

def no_height_obstructions(possible_line,height_gdf, current_supports, plot_possible_lines, ax, location_supports):
    """A function to check whether there are any points along the line candidate (spanned up by the starting/end points elevation plus the support height) which are less than min_height away from the line.

    Args:
        possible_line (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_: _description_
    """    
    support_height = 11
    min_height = 1
    road_height_cutoff = 15
    start_point, end_point = Point(possible_line.coords[0]), Point(possible_line.coords[1])

    # find the elevation of the point in the height gdf closest to the line start point and end point
    max_deviation = 0.1
    start_point_height = fetch_point_elevation(start_point,height_gdf,max_deviation)
    end_point_height = fetch_point_elevation(end_point,height_gdf,max_deviation)

    # add the height of the support to it
    start_point_height+=support_height
    end_point_height+=support_height

    # fetch the floor points along the line
    points_along_line = generate_road_points(possible_line, interval = 1)
    # remove first 15 points since they are on the road and throw off the computation
    #points_along_line = points_along_line[road_height_cutoff:]
    # and their height
    floor_height_below_line_points = [fetch_point_elevation(point,height_gdf,max_deviation) for point in points_along_line]

    # get the distances of each point along the line to the line itself
    # 1. create arrays for start and end point
    line_start_point_array = np.array([start_point.x,start_point.y,start_point_height])
    line_end_point_array = np.array([end_point.x,end_point.y,end_point_height])

    # 2. get the ldh for each point along the line between start and end and put the im an array
    # define variables
    c_rope_length = geometry_utilities.distance_between_3d_points(line_start_point_array,line_end_point_array)
    b_whole_section = start_point.distance(end_point)
    H_t_horizontal_force_tragseil = 80000 #improvised value 
    q_vertical_force = 15000 #improvised value 30kn?
    q_bar_rope_weight = 1.6 #improvised value 2?
    q_delta_weight_difference_pull_rope_weight = 0.6 #improvised value
    # compute distances and create the corresponding points
    ldh_array = np.array([lastdurchhang_at_point(point, start_point, end_point, b_whole_section, H_t_horizontal_force_tragseil, q_vertical_force, c_rope_length, q_bar_rope_weight, q_delta_weight_difference_pull_rope_weight) for point in points_along_line])

    # 3. create an array of the floor points and their distance to the line (without slope)
    floor_points = list(zip([point.x for point in points_along_line], [point.y for point in points_along_line],floor_height_below_line_points))
    line_to_floor_distances = [geometry_utilities.lineseg_dist(point,line_start_point_array, line_end_point_array) for point in floor_points]

    # and finally check the distances between each floor point and the ldh point
    sloped_line_to_floor_distances = line_to_floor_distances - ldh_array

    # return current supports if we are far away enough from the ground
    lowest_point_height = min(sloped_line_to_floor_distances)

    # plot the lines if true
    if plot_possible_lines:
        # plot the failed lines
        ax.plot3D([point[0] for point in floor_points],[point[1] for point in floor_points], floor_height_below_line_points+sloped_line_to_floor_distances)
        #ax.plot3D([start_point.x, end_point.x],[start_point.y, end_point.y],[start_point_height, end_point_height])

    if lowest_point_height>min_height:
        return current_supports, location_supports
    # and enter the next recursive loop if not
    else:
        #1. get the point of contact
        index = int(np.where(sloped_line_to_floor_distances == lowest_point_height)[0])

        #2. "put a support on it" - add support height and flag this as supported
        sloped_line_to_floor_distances[index]+= support_height
        current_supports+=1

        #3. create the new point and lines to/from it
        new_support_point = points_along_line[index]
        road_to_support_line = LineString([start_point, new_support_point])
        support_to_anchor_line = LineString([new_support_point,end_point])

        #4. redo the line height computation left and right recursively
        # add the last argument to prevent adding the supports twice
        current_supports, location_supports = no_height_obstructions(road_to_support_line,height_gdf, current_supports, plot_possible_lines, ax, location_supports)
        current_supports, location_supports = no_height_obstructions(support_to_anchor_line,height_gdf, current_supports, plot_possible_lines, ax, location_supports)
        return current_supports, location_supports.append(floor_points[index])

def lastdurchhang_at_point(point, start_point, end_point, b_whole_section, H_t_horizontal_force_tragseil, q_vertical_force, c_rope_length, q_bar_rope_weight, q_delta_weight_difference_pull_rope_weight):
    """
    Calculates the lastdurchhang value at a given point.

    Args:
    point (Point): The point at which the lastdurchhang value is to be calculated.
    start_point (Point): The start point of the section.
    end_point (Point): The end point of the section.
    b_whole_section (float): The length of the whole section.
    H_t_horizontal_force_tragseil (float): The horizontal force of the tragseil.
    q_vertical_force (float): The vertical force.
    c_rope_length (float): The length of the rope.
    q_bar_rope_weight (float): The weight of the rope.
    q_delta_weight_difference_pull_rope_weight (float): The difference in weight between the pull rope and the tragseil.

    Returns:
    float: The lastdurchhang value at the given point.
    """
    b1_section_1 = start_point.distance(point)
    b2_section_2 = end_point.distance(point)

    lastdurchhang = (b1_section_1*b2_section_2/(b_whole_section*H_t_horizontal_force_tragseil))*(q_vertical_force+(c_rope_length*q_bar_rope_weight/2)+(c_rope_length*q_delta_weight_difference_pull_rope_weight/(4*b_whole_section))*(b2_section_2-b1_section_1))
    return lastdurchhang

def lastdurchhang_mitte():
    q_r_längeneinheitsgewicht_tragseil = 100 #improvised value
    S_bar_t_mittlere_kraft_tragseil = H_t_horizontal_force_tragseil*(c_rope_length/b_whole_section)
    #compute ldh with simplified formula
    ldh_mitte = c_rope_length/4*S_bar_t_mittlere_kraft_tragseil*(q_vertical_force+(c_rope_length*q_r_längeneinheitsgewicht_tragseil)/2)


def fetch_point_elevation(point, height_gdf, max_deviation):
    """
    Fetches the elevation of a given point.

    Args:
    point (Point): The point for which the elevation is to be fetched.
    height_gdf (GeoDataFrame): A GeoDataFrame containing the elevations.
    max_deviation (float): The maximum deviation allowed while fetching the elevation.

    Returns:
    float: The elevation of the given point.
    """
    return height_gdf.loc[(height_gdf.x > point.x-max_deviation) & (height_gdf.x < point.x+max_deviation) & (height_gdf.y < point.y+max_deviation) & (height_gdf.y > point.y-max_deviation),"elev"].values[0]

def generate_support_trees(overall_trees, target, point, possible_line):
    """find trees in overall_trees along the last bit of the possible_line that are close to the line and can serve as support tree

    Args:
        overall_trees (_type_): GDF of all trees
        target (_type_): The last tree
        point (_type_): The road point we are starting from
        possible_line (_type_): The limne between target and point

    Returns:
        _type_: _description_
    """    
    # Parameters
    min_support_sideways_distance = 0.1
    max_support_sideways_distance = 1.5
    min_support_anchor_distance = 10
    max_support_anchor_distance = 20

    # find those trees that are within the sideways distance to the proposed line
    min_support_sideways_distance_trees = overall_trees.geometry.distance(possible_line) > min_support_sideways_distance
    max_support_sideways_distance_trees = overall_trees.geometry.distance(possible_line) < max_support_sideways_distance

    # find those trees that are within the right distance to the target tree
    min_support_anchor_distance_trees = overall_trees.geometry.distance(target) > min_support_anchor_distance
    max_support_anchor_distance_trees = overall_trees.geometry.distance(target) < max_support_anchor_distance

    # shouldnt this be overall_ instead of target_?
    support_tree_candidates = overall_trees[min_support_sideways_distance_trees * max_support_sideways_distance_trees * min_support_anchor_distance_trees * max_support_anchor_distance_trees]

    # select only those support tree candidates which are close to the roadside point than the target tree
    support_tree_candidates = support_tree_candidates[support_tree_candidates.geometry.distance(point) < target.distance(point)]

    return support_tree_candidates


def generate_road_points(road_geometry, interval):
    """Generate a list of points with a given interval along the road geometry

    Args:
        road_geometry (_type_): A list of lines that define the road
        interval: The interval in which a new point is calculated

    Returns:
        _type_: _description_
    """    
    # thanks to https://stackoverflow.com/questions/62990029/how-to-get-equally-spaced-points-on-a-line-in-shapely
    distance_delta = interval
    distances = np.arange(0, road_geometry.length, distance_delta)
    road_points = [road_geometry.interpolate(distance) for distance in distances] + [Point(road_geometry.coords[1])]

    return road_points

def compute_points_covered_by_geometry(points_gdf, geometry, min_trees_covered):
    """Return the points covered by geometry in the points_gdf

    Returns:
        set(geometry), int : set of covered points as well as their amount
    """
    
    contained_points = filter_gdf_by_contained_elements(points_gdf, geometry)

    if len(contained_points) < min_trees_covered:
        return
    # filter only those points
    # and return and set of the covered points as well as the amount of trees covered
    return set(contained_points["id"].values), len(contained_points)


def compute_points_covered_per_row(points_gdf, row_gdf, buffer_size, min_trees_covered):
    """Compute how many points are covered per row.geometry in the points_gdf
    Args:
        points_gdf (_type_): A gdf with a list of point geometries
        row_gdf (_type_): The gdf containing lines where we check how many points are covered
        buffer_size: The width added to the row_gdf.geometry entry
    """

    # already create buffer to avoid having to recreate this object every time
    row_gdf["buffer"] = row_gdf.apply(
        lambda row: row.geometry.buffer(buffer_size), axis=1)

    # appply and return the points covered by each buffer
    return row_gdf["buffer"].apply(lambda row: compute_points_covered_by_geometry(points_gdf, row, min_trees_covered))

def filter_gdf_by_contained_elements(gdf, polygon):
    """Return only the points in the gdf which are covered by the polygon

    Args:
        gdf (_type_): The gdf to filter
        polygon (_type_): A polygon geometry

    Returns:
        _type_: the filtered gdf
    """    
    # get the points which are contained in the geometry
    coverage_series = gdf.geometry.intersects(polygon)
    # and select only those points from the point_gdf
    contained_points = gdf[coverage_series]

    return contained_points

from shapely.ops import nearest_points

def compute_distances_facilities_clients(tree_gdf, line_gdf):
    """Create a numpy matrix with the distance between every tree and line

    Args:
        tree_gdf (_type_): A gdf containing the trees
        line_gdf (_type_): A gdf containing the facilities/lines

    Returns:
        _type_: A numpy matrix of the costs/distances
    """    
    # compute the distance to each tree for every row
    tree_line_distances = []
    carriage_support_distances = []


    for line in line_gdf.iterrows():
        line_tree_distance = tree_gdf.geometry.distance(line[1].geometry)
        # get the nearest point between the tree and the cable road for all trees
        # project(tree,line)) gets the distance of the closest point on the line
        carriage_support_distance = [line[1].geometry.project(Point(tree_geometry.coords[0])) for tree_geometry in tree_gdf.geometry]

        tree_line_distances.append(line_tree_distance)
        carriage_support_distances.append(carriage_support_distance)

    # pivot the table and convert to numpy matrix (solver expects it this way)
    return np.asarray(tree_line_distances).transpose(), np.asarray(carriage_support_distances).transpose()

def generate_triple_angle(point, line_candidate, anchor_trees):
    """Generate a list of line-triples that are within correct angles to the road point and slope line.
    Checks whether:
    - anchor trees are within (less than) correct distance
    - all of those lines have a deviation < max outer anchor angle to the slope line
    Then we create pairs of lines within max_outer_anchor_angle to each other AND one of them with <max_center_tree_slope_angle to the slope line
    This results in pairs of lines were one is the inner and one the right-most (or vice versa)
    Then, we check for a third possible line if it is within 2*min_outer_anch_angle and 2* max_anchor_angle for one point and within min_outer, max_outer for the other -> so two lines are not on the same side!
    This results in triples where we have the central and rightmost line from the tuples section plus one line that is at least 2*min and at most 2*max outer angle away from the rightmost line (or vv.)

    Args:
        point (_type_): The road point we want to check for possible anchors
        line_candidate (_type_): _description_
        anchor_trees (_type_): _description_
        max_anchor_distance (_type_): _description_
        max_outer_anchor_angle (_type_): Max angle between right and left line
        min_outer_anchor_angle (_type_): Minimum angle between right and left line
        max_center_tree_slope_angle (_type_): Max deviation of center line from slope line

    Returns:
        _type_: _description_
    """    
    min_outer_anchor_angle = 20
    max_outer_anchor_angle = 50
    max_center_tree_slope_angle = 3
    max_anchor_distance = 20

    #1. get list of possible anchors -> anchor trees
    anchor_trees_working_copy = deepcopy(anchor_trees)
    
    #2. check which points are within distance
    anchor_trees_working_copy = anchor_trees_working_copy[anchor_trees_working_copy.geometry.distance(point)<max_anchor_distance]

    #3. create lines to all these possible connections
    if anchor_trees_working_copy.empty or len(anchor_trees_working_copy)<3:
        return
    possible_anchor_lines = anchor_trees_working_copy.geometry.apply(lambda x: LineString([x,point]))

    # check if all of those possible lines are within the max deviation to the slope
    possible_anchor_lines = possible_anchor_lines[possible_anchor_lines.apply(lambda x: geometry_utilities.angle_between(x,line_candidate)<max_outer_anchor_angle)].to_list()

    if len(possible_anchor_lines)<3:
        return

    #4. check first pairs: one within 10-30 angle to the other and one should be <5 degrees to the slope line
    pairwise_angle = [(x,y) for x,y in itertools.combinations(possible_anchor_lines,2) 
        if 
            min_outer_anchor_angle < geometry_utilities.angle_between(x,y) < max_outer_anchor_angle 
        and 
            (geometry_utilities.angle_between(x,line_candidate) < max_center_tree_slope_angle 
        or 
            geometry_utilities.angle_between(y,line_candidate) < max_center_tree_slope_angle)]

    # skip if we dont have enough candidates
    if len(pairwise_angle)<3:
        return

    #5. check if the third support line is also within correct angle - within 2*min_outer_anch_angle and 2* max_anchor_angle for one point and within min_outer, max_outer for the other -> so two lines are not on the same side!
    triple_angle = []
    for x,y in pairwise_angle:
        for z in possible_anchor_lines:
            #make sure that we are not comparing a possible line with itself
            if (x is not z and y is not z ):
                a = (max_outer_anchor_angle*2-10 < geometry_utilities.angle_between(x,z)< max_outer_anchor_angle*2 
                and min_outer_anchor_angle < geometry_utilities.angle_between(y,z)< max_outer_anchor_angle)
                b = (max_outer_anchor_angle*2-10 < geometry_utilities.angle_between(y,z)< max_outer_anchor_angle*2 
                and min_outer_anchor_angle < geometry_utilities.angle_between(x,z)< max_outer_anchor_angle)

                if (a,b):
                    triple_angle.append([x,y,z])
    
    return triple_angle