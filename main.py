from gauss_scale_space import *
from cv2 import *
from energy_minimization import *
from component_tree import  *

if __name__ == "__main__":
    img = imread("F:\\line_extraction_python\\101.tif")

    ans = gauss_scale_space(img)
    print('scale spaced finishd')
    pic = ans[0].astype(np.float32)
    map_list = []
    #get components return a list of tuples whch contain a map and number of compnents
    map_list = get_components( pic )
    testmap = map_list[1][0]
    plt.show()
    root_map = map_list[0]
    root = Node(mat_to_index(root_map[0],1))
    make_rec_tree(root,map_list[1:root_map.__len__()])
    potential_lines = find_lines(root)
#    component_list = build_component_list(testmap)
 #   match = make_aribrary_f(component_list,potential_lines)
    plt.imshow( testmap )



    for ans in potential_lines:
        print("Line: ",ans.data)
        xn = [x[0] for x in ans.data]
        yn = [y[1] for y in ans.data]
        plt.scatter( xn, yn, color='red', s=10 )

    plt.show()
    print("END")
