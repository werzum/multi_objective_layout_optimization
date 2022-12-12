from copy import deepcopy
from turtle import distance
from shapely.geometry import LineString, Point

import matplotlib.pyplot as plt

import numpy as np
import math
import itertools

import geometry_utilities

def generate_possible_lines(road_points, target_trees, anchor_trees, overall_trees, slope_line, height_gdf):
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

    max_main_line_slope_deviation = 45

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
                        
                        # and finally check for height obstructions and, if no obstructions were found, append this configuration to the line_gdf
                        if no_height_obstructions(possible_line, height_gdf):
                            angle_between_supports.append(compute_angle_between_supports(possible_line, height_gdf))
                            possible_lines.append(possible_line)
                            slope_deviation.append(possible_line_to_slope_angle)
                            possible_anchor_triples.append(triple_angle)
                            possible_support_trees.append([support_tree_candidates.geometry])

    start_point_dict = {}
    for id,line in enumerate(possible_lines):
        start_point_dict[id] = line.coords[0]

    return possible_lines, slope_deviation, possible_anchor_triples, possible_support_trees, angle_between_supports, start_point_dict

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

def no_height_obstructions(possible_line,height_gdf):
    """A function to check whether there are any points along the line candidate (spanned up by the starting/end points elevation plus the support height) which are less than min_height away from the line.

    Args:
        possible_line (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_: _description_
    """    
    support_height = 8
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
    floor_points = generate_road_points(possible_line, interval = 1)
    # remove first 15 points since they are on the road and throw off the computation
    floor_points = floor_points[road_height_cutoff:]


    #here we add the line computation
        #     onst float Precision = 0.0001;
        # double a_prev = a - IntervalStep;
        # double a_next = a;
        # do
        # {
        #     a = (a_prev + a_next) / 2f;
        #     if (Math.Sqrt(Math.Pow(l, 2) - Math.Pow(v, 2)) < 2 * a * Math.sinh(h/(2*a)))
        #         a_prev = a;
        #     else
        #         a_next = a;
        # } while (a_next - a_prev > Precision);

    #\begin{equation*}\begin{split}h & = x_2 - x_1 \\v & = y_2 - y_1 \\\end{split}\end{equation*}
    # height_difference = x_2-x_1
    # width_difference = y_2-y_1
    # line_length = 100

    # precision = 0.0001
    # a_prev = a - interval_step
    # a_next = a
    # while a_next-a_prev > precision:
    #     a = (a_prev + a_next) / 2f
    #     if (math.sqrt(l**2 - v**2)) < 2 * a * math.sinh(h/(2*a)):
    #         a_prev = a
    #     else:
    #         a_next = a

    # and get their height
    floor_points_height = [fetch_point_elevation(point,height_gdf,max_deviation) for point in floor_points]

    # get the distances of each point along the line to the line itself
    line_start_point_array = np.array([start_point.x,start_point.y,start_point_height])
    line_end_point_array = np.array([end_point.x,end_point.y,end_point_height])
    floor_point_array = list(zip([point.x for point in floor_points], [point.y for point in floor_points],floor_points_height))
    line_to_floor_distances = [geometry_utilities.lineseg_dist(point,line_start_point_array, line_end_point_array) for point in floor_point_array]

    if min(line_to_floor_distances)>min_height:
        return True
    else:
        plt.figure(figsize=(10, 10))
        plt.plot([point.x for point in floor_points], floor_points_height)
        plt.plot([start_point.x, end_point.x],[start_point_height, end_point_height])
        return False


def fetch_point_elevation(point, height_gdf, max_deviation):
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