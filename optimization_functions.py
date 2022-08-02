import numpy as np

#'line_length'      = 0
#'slope_deviation'  = 1
#'amount_trees_covered' = 2, 
#'buffer_width'     = 3, 
#'line_cost'        = 4
#'line_active'      = 5

def line_cost_function(buffer_width, line_length, slope_deviation, line_activated_index):
    return ((buffer_width-10)**2+line_length-30+slope_deviation**2)*line_activated_index


def tree_coverage_function(trees_covered, line_activated_index):
    return -(trees_covered*line_activated_index)
