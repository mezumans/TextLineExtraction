import numpy as np
import component_tree as ctree
from scipy.spatial import distance

class Component( object ):
    def __init__(self, data,id):
            self.data = data
            self.id = id

#gets tuple of first component map and create a list of components
def build_component_list(tup):
    ans= []
    for i in range (1,tup[1]):
        temp = Component( (ctree.mat_to_index( tup[0], i )), i)
        ans.append(temp)
    return ans
#get node and return center of node
def get_centroid(line):
        xn = [x[0] for x in line.data]
        yn = [y[1] for y in line.data]
        length = line.data.__len__()
        sum_x = np.sum(xn)
        sum_y = np.sum( yn )
        return sum_x / length, sum_y / length

#get array and return center of node
def get_centroid_array(array_line):
        xn = [x[0] for x in array_line]
        yn = [y[1] for y in array_line]
        length = array_line.__len__()
        sum_x = np.sum(xn)
        sum_y = np.sum( yn )
        return sum_x / length, sum_y / length

def get_dist(p1,node ):
        return min(distance.cdist( p1, node.data))[0]
#create a list of tuples in which the second object is  a list of nodes and first is a component
def make_aribrary_f(component_list, potential_lines):
        dict= []
        for i in range(1,component_list.__len__()):
                temp_tup = (component_list[i], [potential_lines[i%potential_lines.__len__()]])
                dict.append( temp_tup )

        return dict
