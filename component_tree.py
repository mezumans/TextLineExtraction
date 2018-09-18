import cv2
import numpy as np
from  matplotlib import pyplot as plt
from collections import deque
import queue
class Node(object):
    def __init__(self, data):
        self.data = data
        self.approx = -1
        self.children = []



    def add_child(self, obj):
        self.children.append(obj)

#creating the list of mapped threshold
def get_components(orig_img):
    print('building component tree....')

    ans = []
    pre_tree = []
    s=0

    for i in range(-5,266,10):
       ret_val,temp = cv2.threshold(orig_img,i,255,cv2.THRESH_BINARY)
       temp_int8 = temp.astype(np.int8)

       #print(type(temp[0,0]))
       retval, labels = cv2.connectedComponents( temp_int8 )
       #print( 'map ', i, 'number of components:', retval )
       temp_tup = (labels,retval)
       ans.append( temp_tup )
       #pre_tree.append(labels)

   # tup1 = (pre_tree, ans)
    return ans
#recieve two nodes and returns true if node2 should be a child of node1
def belongs_to(child_comp,father_comp):
    for i in range (1,child_comp.__len__()):
        if ( child_comp[i] not in father_comp ):
            return False
        else :
            return True


#Gets two arguments: source - father component's node, rest -tuple ( list of maps,starting from  next map and the correspondent number of comps)
def make_rec_tree(source,rest):
    if (rest == []):
        return
    next_map = rest[0]
    next_map_compz = []

    for i in range (1, next_map[1]):
        next_map_compz.append(mat_to_index(next_map[0],i))
    for component in next_map_compz :
        if (belongs_to(component,source.data)):
            if(component.__len__ ()== source.data.__len__()):
                make_rec_tree(source, rest[1:rest.__len__()])
            else:
                temp_node = Node(component)
                source.add_child(temp_node)

                make_rec_tree(temp_node, rest[1:rest.__len__()])

"""
def build_tree(components_array):
    maps = components_array[0]
    num_of_comp = components_array[1]
    length = maps.__len__()
    root_data = mat_to_index(maps[0],1)
    root = Node(root_data)


    for i in range(1,length):
        num_of_iterations = num_of_comp[i]
        curr_mat = maps[i]
        next_mat = maps[i + 1]


        for k in range (1,num_of_iterations):
            indexs_curr_mat = mat_to_index( curr_mat,k )
            return tree
"""


        # retval, labels = cv2.connectedComponents(ans[j])
    #The func gets the connected component array

#building the tree
     #for each threshold
        #Iterating the components and check in the i+1 if there are subcomponents
        #add them to the tree (add the indexes to the node as a list

#get matrix and component and return the indexes of the co
def mat_to_index(mat,component):
    ans = []

   # plt.imshow(mat)
   # plt.show()
    img_size = mat.shape
    rows = img_size[0]
    cols = img_size[1]

    for i in range(0, rows):
        for j in range(0, cols):
            if(mat[i,j] == component):
                tup = (j,i)
                ans.append(tup)

    return ans
#determine threshold to know ifline
#iterate the resulted tree and foreach node make linear piecewise approximation(polyfit foreach node)
#compare the between polfit(x) and y
#if the result of the comparison > thershold (represent a line that we defined) we saves the index

def test_make_tree_maps(maps_list,root, depth):
    if (maps_list == []):
        print("tests amazing!!")
        return True
    temp_compz = []
    for i in range(0,maps_list[0][1]):
        temp_compz.append(mat_to_index(maps_list[0][0],i))

    for child in root.children:
        if child.data not in temp_compz:
            print("TEST failed, depth:" , depth, "number of children:", root.children.__len__())

            return False
    for child in root.children:
        test_make_tree_maps(maps_list[1:maps_list.__len__()], child, depth+1)




#Gets the tree and apply polyfit on eache node.
#returns a value which indicates if it's a line or not
def calc_fit(node):
    xn = [x[0] for x in node.data]
    yn = [y[1] for y in node.data]
    linear_eq = np.polyfit( xn, yn, 1 )
    yn_hat = np.polyval( linear_eq, xn )
    fit = np.linalg.norm( yn_hat - yn, 1 ) / xn.__len__()
    return fit

"""
    weights = 1
    f = np.poly1d( [-5, 1, 3] )
    x = np.linspace( 0, 2, 20 )
    x= np.ndarray()

    y = f( x ) + 1.5 * np.random.normal( size=len( x ) )
    xn = np.linspace( 0, 2, 200 )
    popt = np.polyfit(x, y, 2)



    chi2 = np.sum(weights*(p(x) - y)**2)
"""
#IMALe
#func returns a list of nodes that are liens
THRESHOLD = 10

def find_lines(root):
    counter = 0
    output = []
    q = queue.Queue()
    q.put(root)
    while not q.empty():
        c = q.get()
        if (calc_fit(c) < THRESHOLD):
            print (calc_fit(c))
            counter+=1
            output.append(c)
        else:
            for child in c.children:
                q.put(child)

    print('total amount of lines: ', counter)
    return output