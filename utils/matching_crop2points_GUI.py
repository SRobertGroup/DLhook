
import os
import cv2 

#File formate
# c_points=[(1590, 1267), (3205, 1296)]
# start_p=[(2920, 1501), (1819, 1448)]

#This class sorts the selected starting points from the GUI its respective crop-image
class match_crop_points:
    def __init__(self, crop_mid_points, seedling_starring_points):

        #Sort the points first from left to right, the points are then splitted into a upper and a lower region
        seedling_starring_points =sorted(seedling_starring_points, key=lambda tup: tup[0])
        seedling_starring_points =sorted(seedling_starring_points, key=lambda tup: tup[1])
        
        self.matched_starting_points=[[] for n in range(len(crop_mid_points))]
        self.ratio_points=[[] for n in range(len(crop_mid_points))]
        self.ratio_points=[[] for n in range(len(crop_mid_points))]

        for point_x in seedling_starring_points:
            """
            The match variable is a check so that the staring points only gets added to one crop
            """
            match=0
            x = point_x[0]
            y = point_x[1]
            for n, crop_point in enumerate(crop_mid_points):
                x1=crop_point[0]-512
                x2=crop_point[0]+511
                y1=crop_point[1]-512
                y2=crop_point[1]+511
                if (x in range(x1, x2)) and (y in range(y1, y2)):
                    self.ratio_points[n].append((x-x1, y-y1))
                    self.matched_starting_points[n].append((x, y))

                    match=1
                if match==1:
                    break


    def return_crop_points(self):

        return self.ratio_points
