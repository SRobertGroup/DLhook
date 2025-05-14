
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io
import matplotlib.image as mpimg



path_pre='data/predict/'

"""
input = [[(x1,y2),(x2,y2)],[(x3,y3)]]
output= [[1,2],[3]]

This class takes lists of tuples as input containing the seedling starting points (x,y) and returns a
a list contating in the same structure as the input which gives each point a specific nunber/id
"""
class point_num():
    def __init__(self,list1):
        self.list1=list1
    def point_numbering(self):
        numb=[]
        
        n=1
        for crop in self.list1:
            
            n_x=len(crop)
            n_x=n+n_x
            list_x=[n for n in range(n,n_x)]
            n = n_x
            numb.append(list_x)
        return numb
                    

"""
This class postprocess the predicted masks
1: Takes away the lower cotyledon - remove cotyl under the set starting point by the user.
2: Removes hypocotyl which are located above the highest cotyledon in the image
"""
class postprocess_masks():
    def __init__(self,seed_coat_points):

        for n in range(len(seed_coat_points)):
            img_paths_n=self._image_crops_path(n)
            postprocess_masks=self._generate_mask(img_paths_n, seed_coat_points[n])
    def _image_crops_path(self, id_n):
        path_pre='data/predict/'
        img_cotyl_paths=[path_pre+img for img in os.listdir(path_pre) if (int(img[-5]) == 1) and  (int(img[0]) == id_n)]
        img_hypocotyl_paths=[path_pre+img for img in os.listdir(path_pre) if (int(img[-5]) == 2) and (int(img[0]) == id_n)]
        image_paths=(img_cotyl_paths, img_hypocotyl_paths)
        return image_paths

    def _generate_mask(self, img_paths,seed_coat_points):
        if len(seed_coat_points)==0:
            return
        path_post4='data/postprocess/'

        img_cotyl_paths2, img_hypocotyl_paths2 = img_paths
        cotyl_cont=[]

        #This section creates a mask which will remove the hypocotyl(root-junction) which are located under the cotyledons
        #The points(seed starting point) are given by the user.
        print("seed_coat_points",seed_coat_points)
        mask_region=seed_coat_points
        x_1=1024
        y_1=seed_coat_points[-1][1]
        mask_region.append([x_1, y_1])
        x_2=1024
        y_2=1024
        mask_region.append([x_2, y_2])

        x_3=1
        y_3=1024
        mask_region.append([x_3, y_3])
        x_last=1
        y_last=seed_coat_points[0][1]


        mask_region.append([x_last, y_last])
        mask = np.array(mask_region, np.int32)


        mask_region_lower_cotyl = seed_coat_points
        mask_region_lower_cotyl = np.array(mask, np.int32)



        for idx, img_path in enumerate(img_hypocotyl_paths2):
            hypo_name=img_path.split('/')[-1]

            cotyl_name=img_cotyl_paths2[idx].split('/')[-1]
        
            #Lower region Hypocotyl mask 
            img_x = cv2.imread(img_path, 0)
            img_x_modified= cv2.fillConvexPoly(img_x, mask, 255)
            img_x_modified1=img_x_modified        
            mask_2=mask_region_lower_cotyl

            img_seed_coat=cv2.imread(img_cotyl_paths2[idx], 0)
            img_x_modified= cv2.fillConvexPoly(img_seed_coat, mask_2, 255)
            img_x_modified= cv2.bitwise_not(img_x_modified)


            """
            The following section is for the cotyledon postprocessing
            """
            kernel= np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(img_x_modified, cv2.MORPH_OPEN, kernel,iterations=3)
            opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,kernel, iterations=3)
            opening = cv2.erode(opening,kernel,iterations = 2)
            #dilation = cv2.dilate(opening, kernel, iterations=3)


            opening = cv2.bitwise_not(opening)
            ret1, thresh = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            

            """
            The following section finds the cotyledon located at the highest point on the image
            and removes all of the hypocotyl regions which are located above the point

            """
            y_min=1024
            for contour_x in contours:
                if len(contour_x) >=5:
                    ellipse_x = cv2.fitEllipse(contour_x)
                    y_point = int(ellipse_x[0][1])
                    if y_point < y_min:
                        y_min = y_point

            cotyl_cont.append(contours)

            img_x_modified1[0:y_min, 1:1024]=255
            img_x_modified1= cv2.fillConvexPoly(img_x_modified1, mask_2, 255)
            img_x_modified1=cv2.bitwise_not(img_x_modified1)
            kernel = np.ones((3,3),np.uint8)


            """
            The following section is for the hypocotyl postprocessing
            """
            img_x_modified1 = cv2.morphologyEx(img_x_modified1, cv2.MORPH_OPEN, kernel,3)

            img_x_modified1 = cv2.morphologyEx(img_x_modified1, cv2.MORPH_CLOSE, kernel)
            img_x_modified1 = cv2.morphologyEx(img_x_modified1, cv2.MORPH_CLOSE, kernel)
            #img_x_modified1 = cv2.dilate(img_x_modified1,kernel,iterations = 5)
            img_x_modified1 = cv2.erode(img_x_modified1,kernel,iterations = 2)
            img_x_modified1=cv2.bitwise_not(img_x_modified1)

            cv2.imwrite(path_post4+hypo_name, img_x_modified1)
            cv2.imwrite(path_post4+cotyl_name, opening)
