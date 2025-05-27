import os
from typing_extensions import final
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io
import matplotlib.image as mpimg
import math
from utils.postprocess_model_output import *

class apical_hook():
    def __init__(self, img_name, points,point_numbering, previous_angles, previous_angles_max):
        self.save_filename=img_name
        self.output_image = cv2.imread('data/images/'+img_name,1)
        self.selected_points = points[:-4]
        self.point_index = point_numbering

        """
        previous_angles is the angles from the previous timestep, the angles are used when measuring the
        angles for the next image.

        """
        self.previous_angles=previous_angles

        """
        previous_angles_max is a dictionary containing the maximum anlge-value each of the seedlings has had 
        at any timepoint during the kinematics, these values are used in order for the software to calculate the 
        correct angle for the next timepoints.
        """
        self.previous_angles_max=previous_angles_max


        
        
        path_img='data/postprocess/'
        img_path_cotyl=([path_img+img_name[:-4]+'-1.png'])
        img_path_hypo=([path_img+img_name[:-4]+'-2.png'])

        self.matched_points_images=[]

        self._cotyl_contours=[]
        self._hypo_contours=[]

        self._list_array_cotyl=[]
        self._list_array_hypo=[]

        #Find the contours of the cotyls
        for img_path_x in img_path_cotyl:
            img = cv2.imread(img_path_x, 0)
            ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Ensure it's a list
            contours = list(contours)
            # Filter out short contours safely
            contours = [cnt for cnt in contours if len(cnt) >= 5]
            # for i,contour_x in enumerate(contours):
            #     if len(contour_x)<5:
            #         del contours[i]


            self._cotyl_contours.append(contours)
            array_cotyl = np.empty((len(contours), 1))
            self._list_array_cotyl.append(array_cotyl)


        for img_path_x in img_path_hypo:
            img = cv2.imread(img_path_x, 0)
            ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(contours)
            contours = [cnt for cnt in contours if len(cnt) >= 5] # 26/05
            # for i,contour_x in enumerate(contours):
            #     if len(contour_x)<=10:
            #         del contours[i]

            self._hypo_contours.append(contours)
            array_hypo = np.empty((len(contours), 1))
            self._list_array_hypo.append(array_hypo)
        
        #The following rows are matching the cotyledon to the closests hypocotyl with different sized squares around the cotyledons
        self.matched_points_list=[]
        self._combine_cotyl_hypo(6)
        self.matched_points_list.append(self.matched_plant)
        self._combine_cotyl_hypo(10)
        self.matched_points_list.append(self.matched_plant)
        self._combine_cotyl_hypo(15)
        self.matched_points_list.append(self.matched_plant)
        self._combine_cotyl_hypo(20)
        self.matched_points_list.append(self.matched_plant)
        self._combine_cotyl_hypo(30)
        self.matched_points_list.append(self.matched_plant)


        self.matched_plant=self._check_common_cotyl(self.matched_points_list)




    def _combine_cotyl_hypo(self, area_cotyl):
        #For every image crop
        for id_img, _ in enumerate(self._hypo_contours):
            self.total_ellips_cotyl=[]
            self.stem_cont_p=[]
            self.cotyl_cont_p=[]
            self.stem_real_points=[]
            self.combine_list = []

            for n, _ in enumerate(self._hypo_contours[id_img]):
                self.combine_list.append(([], []))
                self.stem_real_points.append([])

            for n, stem_n in enumerate(self._hypo_contours[id_img]):
                for point in range(len(stem_n)):
                    x_p = stem_n[point][0][0]
                    y_p = stem_n[point][0][1]
                    self.stem_real_points[n].append((x_p, y_p))

            self.cotyl_rect = []
            self.cotyl_rect_2 = []
            self.cotyl_rect_3 = []
            self.cotyl_rect_4 = []
            self.cotyl_cent = []

            for cotyledon_n in self._cotyl_contours[id_img]:
                if len(cotyledon_n) >=5:

                    self.ellipse_cotyl = cv2.fitEllipse(cotyledon_n)

                    self.total_ellips_cotyl.append(self.ellipse_cotyl)

                    X_point, Y_point = int(self.ellipse_cotyl[0][0]), int(self.ellipse_cotyl[0][1])

                    self.cotyl_cent.append((X_point, Y_point))

                    #This section creates an area around each cotyledon to be matched with the closest hypocotyl
                    self.cotyl_rect.append(self._rectangle(X_point, Y_point, area_cotyl))

            

            self.arr_stem_points = np.empty(len(self.stem_real_points), dtype=object)


            self.arr_cot_rect = np.empty(len(self.cotyl_rect), dtype=object)
            self.arr_cot_rect_2=np.empty(len(self.cotyl_rect_2), dtype=object)
            self.arr_cot_rect_3=np.empty(len(self.cotyl_rect_3), dtype=object)
            self.arr_cot_rect_4=np.empty(len(self.cotyl_rect_4), dtype=object)

            arrays_cot=[self.arr_cot_rect,self.arr_cot_rect_2,self.arr_cot_rect_3,self.arr_cot_rect_4]

            self.matched_plant = []

            # Make arrays for x,y points of contours of the stems
            for ind1 in range(len(self.arr_stem_points)):
                stem_x = self.stem_real_points[ind1]
                out = np.empty(len(stem_x), dtype=object)
                out[:] = stem_x
                out1 = np.stack(out)

                self.arr_stem_points[ind1] = out1


            # Make arrays for points in rectangle around cotyledons
            for ind2 in range(len(self.arr_cot_rect)):
                cot_x = self.cotyl_rect[ind2]
                out2 = np.empty(len(cot_x), dtype=object)
                out2[:] = cot_x
                out22 = np.stack(out2)
                self.arr_cot_rect[ind2] = out22

            # Comparing to see if (x,y) points are in both stem and cotyl box if so I match them
            """
            The following loops checks for matching hypocotyl-cotyledons. It appends the indexes of 
            (index-stem, index-cotyl) to matched_plant
            """
            for j in range(len(self.arr_stem_points)):
                stem_1 = self.arr_stem_points[j]
                for j2 in range(len(self.arr_cot_rect)):

                    cotyl_1 = self.arr_cot_rect[j2]
                    # Compare elementsrange(len(filter_comb)):
                    check2 = (stem_1[:, None, :] == cotyl_1[None, :, :]).all(axis=-1)
                    if True in check2:
                        self.matched_plant.append((j, j2))

    def _check_common_cotyl(self,list_matched_points):
        matched_points_new=[]
        cotyls_present=[]
        for match_point_x in list_matched_points:
            for tuples_x in match_point_x:
                cotyl_point=tuples_x[1]
                if cotyl_point not in cotyls_present:
                    cotyls_present.append(cotyl_point)
                    matched_points_new.append(tuples_x)

        return matched_points_new



    def list_nparray(self,list1):
        self.empty_arr=np.empty(len(list1), dtype=object)

        for n in range(len(self.empty_arr)):

            rect11 = list1[n]
            out = np.empty(len(rect11), dtype=object)
            out[:] = rect11
            out1 = np.stack(out)

            self.empty_arr[n] = out1
        return(self.empty_arr)

    def _rectangle(self, x2, y2, radius):
        self.match=[]

        for row1 in range((x2-radius),(x2+radius)):
            for column1 in range((y2+5),(y2+radius)):
                self.match.append((row1, column1))
        return (self.match)

    def _rectangle_crop(self, x2, y2, radius):
        self.match=[]

        for row1 in range((x2-3),(x2+3)):
            for column1 in range((y2),(y2+radius)):
                self.match.append((row1, column1))
        return (self.match)
    


    """
    The following function _rectangle_long generates a rectangle from the starting points which will be matched with the point.
    This was done as some seedlings might germinate in later stages, which shows the order-id of the seedling. Thus the starting poins are given by the user
    """
    def _rectangle_long(self, x , y):
        self.long_rect=[]
        # the values in row determines how wide the rectangle from the starting point will be
        for row in range((x-38),(x+38)):
            for col in range((y-150),y):
                self.long_rect.append((row ,col))

        return(self.long_rect)




    """
    The following function looks for stems with multiple cotyledons
         
    """
    def angles(self):
        color=(215,138,94)
        #Original image which will be used to write over
        # self.self.output_image = cv2.imread('data/images/'+self.save_filename,1)

        filter_comb = []
        """ 
        Remove the cotyledons which are located on the lower region of hypocotyl, if two cotyledons are connected to the hypocotyl
        """
        for i1 in range(len(self._hypo_contours[0])):
            paired_to_stem = [item for item in self.matched_plant if item[0] == i1]

            length1 = len(paired_to_stem)
            if length1 > 1:
                coordinates = []
                pair = []

                for stemX in paired_to_stem:

                    coordinates.append(self.cotyl_cent[stemX[1]])
                    pair.append(stemX)

                if (coordinates[0][1] - coordinates[1][1]) < 0:
                    filter_comb.append(pair[0])
                else:
                    filter_comb.append(pair[1])
            if length1 == 1:
                filter_comb.append(paired_to_stem[0])



        # Create matrix with the combined data of matching hypocotyl-cotyledon, x-,y-points of center cotyledon.
        combined_data = np.empty((len(filter_comb), 8), dtype=object)
        for match_ID in range(len(filter_comb)):
            (stem_ID, cotyl_ID) = filter_comb[match_ID]
            (x_val, y_val) = self.cotyl_cent[cotyl_ID]
            combined_data[match_ID][0] = (stem_ID, cotyl_ID)
            combined_data[match_ID][1] = x_val
            combined_data[match_ID][2] = y_val
            # combined_data[match_ID][3]

        # Sort the data by the value in the X column so it numbers the apical hook from left to right.
        ind = np.argsort(combined_data[:, 1])
        sort_data = combined_data[ind]


        # Calculate the angle of the stem with ellipse of stem
        # Creating a big rectangle where the points of the stem should be included in when calculating the angle.
        # radius 65
        
        crop_region_hypo_top=75

        for n2 in range(len(filter_comb)):
            ID_ellipse = sort_data[n2][0][1]
            x_1 = sort_data[n2][1]
            y_1 = sort_data[n2][2]

            rect1 = self._rectangle(x_1, y_1, crop_region_hypo_top)
            array1 = np.empty(len(rect1), dtype=object)
            array1[:] = rect1
            rect1 = np.stack(array1)
            sort_data[n2][3] = rect1
            sort_data[n2][4] = self.total_ellips_cotyl[ID_ellipse]


        stem_ellipse = []

        # For every combined object (stem and cotyledon) find the common points from stem and cotyledon rectangle
        n3 = 0
        for point in sort_data:

            common_points = []

            for element in self.arr_stem_points[point[0][0]]:
                check5 = any(np.equal(point[3], element).all(1))

                if check5 == True:
                    common_points.append((element[0], element[1]))

            stem_ellipse.append(common_points)
            s4 = np.asarray(common_points, dtype=np.int32)
            
            if (len(s4) >= 5):
                s5 = cv2.fitEllipse(s4)
                sort_data[n3][5] = s5
                n3 = n3 + 1

        # Find the common points and draw ellipse for each trimmed stem
        self.sort_data = sort_data
        stem_angles=[]
        cotyl_angles=[]
        stem_angles2=[]
        cotyl_angles2=[]

        seedling_id=[]
        seedling_angles=[]

        #Generates rectangles above starting points to match the hypocotyl with.
        list_l_rect = []
        for s_point in self.selected_points:
            l11 = (self._rectangle_long(s_point[0], s_point[1]))

            list_l_rect.append(l11)


        r_array = self.list_nparray(list_l_rect)
        self.r_array = r_array

        #This json file will contain the debug data if debug mode is turned on,
        image_metadata_json={}

        #Loop through all seedlings in a image-patch/crop
        for seedling_n in range(len(sort_data)):

            stem_x = self.arr_stem_points[sort_data[seedling_n][0][0]]
            self.stem_x = stem_x

            """
            Loops through the metadata for all of the seedlings. This data is used calculate the apical hook angle
            """
            for start_point_rect_n in range(len(r_array)):
                
                check3 = (stem_x[:, None, :] == r_array[start_point_rect_n][None, :, :]).all(axis=-1)
                
                #If match between staring points and a seedling(cotyl-hypocotyl point)
                if True in check3:

                    element1=sort_data[seedling_n]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 2
                    # Blue color in BGR

                    #Colors of output image lines and dots
                    color_line_cotyledon = (215,138,94)
                    color_line_hypocotyl = (219,107,98)
                    color_numbers=(113,131,168)




                    # Line thickness of 2 px
                    thickness = 1
                    # Using cv2.putText() method

                    cotyl_el = element1[4]
                    if cotyl_el is None:
                        continue
                    c_length = cotyl_el[1][1]
                    c_ang = cotyl_el[2]
                    c_ang1=c_ang
                    c_ang_line=c_ang


                    c_points = cotyl_el[0]
            

                    mid_x_cotyl = int(c_points[0])
                    mid_y_cotyl = int(c_points[1])



                    c_ang = self._correct_angle(c_ang)
                    cotyl_angles.append(c_ang)

                    angle = (c_ang * math.pi) / 180
                    cotyl_angles2.append(angle)
                    rc = c_length / 2



                    stem_el = element1[5]
                    if stem_el is None:
                        continue
                    s_length = stem_el[1][1]
                    s_ang = stem_el[2]

                    s_correct_angle=self._correct_angle(s_ang)
                    angle2=(s_correct_angle * math.pi) / 180

                    s_ang2=s_ang
                    s_ang_line=s_ang

                
                    s_points = stem_el[0]
                    (d1, d2)= stem_el[1]
                    rmajor = max(d1,d2)/2
                    rmajor=45
                    if s_ang_line > 90:
                        s_ang_line = s_ang_line - 90
                    else:
                        s_ang_line = s_ang_line + 90


                    x_c_hypo = int(s_points[0])
                    y_c_hypo = int(s_points[1])

                    s_x=x_c_hypo

                    #Hypocotyl line points
                    rc=22
                    rs=45
                    rem=20
                    rem2=40

                    """
                    The following section is plots the lines that determines the angles of the 
                    hypocotyl and cotyledon. 
                    """


                    #Cotyledon line points
                    x_top_cot, y_top_cot = int(mid_x_cotyl + rc * math.cos(angle)), int(mid_y_cotyl + rc * math.sin(angle))
                    x_bot_cot, y_bot_cot = int(mid_x_cotyl - rc * math.cos(angle)), int(mid_y_cotyl - rc * math.sin(angle))

                    #put cotyledon line 
                    cv2.line(self.output_image, (x_top_cot, y_top_cot), (x_bot_cot, y_bot_cot), color_line_cotyledon, 2)
                    x_top_hypo, y_top_hypo = int(x_c_hypo + rs * math.cos(angle2)), int((y_c_hypo+rem) + rs * math.sin(angle2))
                    x_bot_hypo, y_bot_hypo = int(x_c_hypo - rs * math.cos(angle2)), int((y_c_hypo+rem) - rs * math.sin(angle2))
                    
                    """
                    
                    I want to check the difference x in cotyledon and hypocotyl 
                    if the diff is too large then I will adjust the 
                    """
                    def _check_x_diff(self, x_center_cotyl, x_center_hypo):
                        if abs(x_center_cotyl-x_center_hypo)>15:
                            new_x_center=abs(x_center_cotyl-x_center_hypo)/2
                            return new_x_center
                        else:
                            return x_center_cotyl

                    """
                    check which side 
                    
                    """
                    if abs(mid_x_cotyl-x_c_hypo)>26:

                        x_shift=8                      
                        if abs(mid_x_cotyl-x_c_hypo)<0:
                            x_c_hypo=mid_x_cotyl+x_shift
                        else:
                            x_c_hypo=mid_x_cotyl-x_shift


                        

                    
                    if (y_bot_hypo-y_top_cot)<0:
                        x_top_hypo, y_top_hypo = int(x_c_hypo + rc * math.cos(angle2)), int((y_bot_cot+rem2) + rc * math.sin(angle2))
                        x_bot_hypo, y_bot_hypo = int(x_c_hypo - rc * math.cos(angle2)), int((y_bot_cot+rem2) - rc * math.sin(angle2))



                    else:

                        x_top_hypo, y_top_hypo = int(x_c_hypo + rc * math.cos(angle2)), int((y_bot_cot+rem2) + rc * math.sin(angle2))
                        x_bot_hypo, y_bot_hypo = int(x_c_hypo - rc * math.cos(angle2)), int((y_bot_cot+rem2) - rc * math.sin(angle2))


                    cv2.line(self.output_image, (int(x_top_hypo),int(y_top_hypo)), (int(x_bot_hypo),int(y_bot_hypo)), color_line_hypocotyl, 2)


                    midx = int(s_points[0])
                    midy = int(s_points[1])

                    s_ang = self._correct_angle(s_ang)
                    stem_angles.append(s_ang)

                    angle = (s_ang * math.pi) / 180
                    stem_angles2.append(angle)
                    r = s_length / 2
                    r=45
                    x1, y1 = int(midx + r * math.cos(angle)), int(midy + r * math.sin(angle))
                    x2, y2 = int(midx - r * math.cos(angle)), int(midy - r * math.sin(angle))

                    return_angle_x,debug_data=self._calculate_angle_final(self.point_index[start_point_rect_n],c_ang1,s_ang2, (int(x_bot_hypo),int(y_bot_hypo)), (int(x_c_hypo),int(y_c_hypo)), (x_bot_cot, y_bot_cot), (x_top_cot, y_top_cot))

                    image_metadata_json[f"seedling-ID:{self.point_index[start_point_rect_n]}"]=debug_data


                    seedling_id.append(self.point_index[start_point_rect_n])
                    seedling_angles.append(return_angle_x)

                    x1=self.selected_points[start_point_rect_n][0]
                    y1=self.selected_points[start_point_rect_n][1]
                    cv2.putText(self.output_image, str(self.point_index[start_point_rect_n]), (x1, y1), font, 1, color_numbers, 3,
                                cv2.LINE_AA)

        cv2.imwrite('data/final_prediction/'+self.save_filename,self.output_image)



        return seedling_angles,seedling_id,image_metadata_json

    #Transforms the angle from fit ellipse into the correct form
    def _correct_angle(self, angle):
        angle=round(angle)
        if angle>90:
            angle=angle-90
        else:
            angle=angle+90
        return angle


    
    """
    The function _calculate_angle_final contains the logic for  calculating the apical hook angle based on the angles from OpenCV fitelipse. 
    """
    def _calculate_angle_final(self,seedling_id, angle_c, angle_s, s_top,s_mid, c_points_upper, c_points_bottom):

        stem_x, stem_y= s_top
        stem_m_x, stem_m_y=s_mid
        cot_x_u, cot_y_u = c_points_upper
        cot_x_b, cot_y_b= c_points_bottom

        #Check in which way the cotyledon is pointing
        check_1_cotyl=stem_x-cot_x_u
        check_2_cotyl=stem_x-cot_x_b

        font = cv2.FONT_HERSHEY_SIMPLEX
        first_image=False
        if type(self.previous_angles)==dict:
            
            previous_angle=self.previous_angles.get(seedling_id)
        else:
            first_image=True

        if type(self.previous_angles_max)==dict:
            previous_angle_max=self.previous_angles_max.get(seedling_id)


        #Check if overhook is present 
        overhook=False
        if abs(cot_x_u-stem_m_x)>abs(cot_x_b-stem_m_x):
            overhook=True
         

        #Overhook check
        overhook_=0

        if abs(stem_x-cot_x_u)>abs(stem_m_x-cot_x_b):

            overhook_=1

        angle_over=0

        #Check if upper cot point is further away from hypocotyl then lower point
        if abs(check_1_cotyl)>abs(check_2_cotyl):
            #Upper point further away
            direction=check_1_cotyl
            if type(self.previous_angles_max)==dict:
                if previous_angle_max:
                    if previous_angle_max>110:
                        angle_over=1

        """
        Check if cotyledon is pointing upwards or downwards 
        by first finding the closes point which is "connected" to the upper part of the hypocotyl

        If the point that is not "connected" is located at a higher Y-point than the "connected"-Y-point 
        then the cotyledon is pointing upwards 

        """
        cot_upper_check=math.sqrt((stem_x-cot_x_u)**2+((stem_y+3)-cot_y_u)**2)
        cot_bottom_check=math.sqrt((stem_x-cot_x_b)**2+((stem_y+3)-cot_y_b)**2)
        
        
        pointing=0

        #If upper point of cotyledon is connected to hypocotyl then the cotyledon is pointing downwards
        if cot_upper_check>cot_bottom_check:
            pointing=1

        #Check if lower cot point is further away from hypocotyl then upper point
        if abs(check_1_cotyl)<abs(check_2_cotyl):
            #Lower point further away

            angle_over=0
            direction=check_2_cotyl
        if abs(check_1_cotyl)==abs(check_2_cotyl):

            direction=check_1_cotyl
            angle_over=1

        
        if direction<0:
            #Pointing Right
            # cv2.circle(self.self.output_image,(stem_x,stem_y), 3, (255,255,255), 2)
            dir_c=0
        if direction>=0:
            #Pointing Left
            # cv2.circle(self.self.output_image,(stem_x,stem_y), 3, (0,0,0), 2)
            dir_c=1

        #calc closest point

        #These variables are used to check which type of angle orientation is present, clockwise/anticlock. 
        cot_angle_check=0
        stem_angle_check=0
        if 90<angle_c<180:
            # angle_c=180-angle_c
            cot_angle_check=1
            # angle_c=180-angle_c
        if 90<angle_s<180:
            # angle_c=180-angle_s
            stem_angle_check=1
            # angle_s=180-angle_s


        """
        dir_c = 0 pointing right WHITE
        dir_c = 1 pointing left BLACK
        """
        if (cot_angle_check==1 and stem_angle_check==1):
            #Right - White
            if dir_c==0:
                final_angle=abs(angle_s-angle_c)
                final_angle_code='final_angle=abs(angle_s-angle_c)'
                final_angle_calc=f'final_angle=abs({int(angle_s)}-{int(angle_c)})'

                if type(self.previous_angles)==dict:
                    if previous_angle and final_angle and previous_angle_max:
                        if previous_angle_max>110 and pointing==1:
                            final_angle=180-abs(angle_s-angle_c)
                            final_angle_code='final_angle=180-abs(angle_s-angle_c)'
                            final_angle_calc=f'final_angle=180-abs({int(angle_s)}-{int(angle_c)})'

                if final_angle==180:
                    if type(self.previous_angles_max)==dict:
                        if previous_angle_max:
                            if previous_angle_max<120:
                                final_angle=0
                                final_angle_code='final_angle=0'
                                final_angle_calc='final_angle=0'
            #Left - Black
            if dir_c==1:

                final_angle=abs(angle_s-angle_c)
                final_angle_code='final_angle=abs(angle_s-angle_c)'
                final_angle_calc=f'final_angle=abs({int(angle_s)}-{int(angle_c)})'
 

                if type(self.previous_angles)==dict:
                    if previous_angle and final_angle and previous_angle_max:
                        if previous_angle_max>80:
                            final_angle=180-abs(angle_s-angle_c)
                            final_angle_code='final_angle=180-abs(angle_s-angle_c)'
                            final_angle_calc=f'final_angle=180-abs({int(angle_s)}-{int(angle_c)})'


        if (cot_angle_check==0 and stem_angle_check==0):
            # Left - White 
            if dir_c==1:
                final_angle=abs(angle_s-angle_c)
                final_angle_code='final_angle=abs(angle_s-angle_c)'
                final_angle_calc=f'final_angle=abs({int(angle_s)}-{int(angle_c)})'



            # Right - Black 
            if dir_c==0:
                #final_angle=(180-angle_c)+angle_s
                final_angle=abs(angle_s+angle_c)
                final_angle_code='final_angle=abs(angle_s+angle_c)'
                final_angle_calc=f'final_angle=abs({int(angle_s)}+{int(angle_c)})'
                
                if type(self.previous_angles)==dict:
                    if previous_angle and final_angle and previous_angle_max:
                        if previous_angle_max>=90:
                            final_angle=180-abs(angle_s-angle_c)
                            final_angle_code='final_angle=180-abs(angle_s-angle_c)'
                            final_angle_calc=f'final_angle=(180-{int(angle_c)})+{int(angle_s)})'

        if (cot_angle_check==1 and stem_angle_check==0):
            # Right - White 
            if dir_c==0:
                    if angle_c==180:
                        angle_c=0
                    final_angle=((180-angle_c)+angle_s)
                    final_angle_code='final_angle=((180-angle_c)+angle_s)'
                    final_angle_calc=f'final_angle=(180-{int(angle_c)})+{int(angle_s)})'
                #final_angle=angle_s-angle_c
            # Left - Black 
            if dir_c==1:
                if angle_c==180:
                    angle_c=0
                final_angle=(180-angle_c)+angle_s
                final_angle_code='final_angle=(180-angle_c)+angle_s'
                final_angle_calc=f'final_angle=(180-{int(angle_c)})+{int(angle_s)})'


                if angle_s==0:
                    final_angle=angle_c
                if type(self.previous_angles_max)==dict:
                        if previous_angle_max:
                            if pointing==1 and previous_angle_max>60:

                                final_angle=(angle_c-angle_s)
                                final_angle_code='final_angle=(angle_c-angle_s)'
                                final_angle_calc=f'final_angle=({int(angle_c)}-{int(angle_s)})'

                #final_angle=angle_c-angle_s

        if (cot_angle_check==0 and stem_angle_check==1):
            # Right - White 
            if dir_c==0:
                # final_angle=180-((180-angle_s)+angle_c)
                final_angle=((180-angle_s)+angle_c)
                final_angle_code='final_angle=((180-angle_s)+angle_c)'
                final_angle_calc=f'final_angle=((180-{int(angle_s)})+{int(angle_c)})'

            # Left - Black 
            if dir_c==1:
                if angle_c==180:
                    angle_C=0
                final_angle=(180-angle_s)+angle_c
                final_angle_code='final_angle=(180-angle_s)+angle_c'
                final_angle_calc=f'final_angle=(180-{int(angle_s)})+{int(angle_c)}'



        final_angle=round(final_angle)

        

        if first_image and overhook_==1:

            if final_angle>90:
                final_angle=-(180-final_angle)
                final_angle_code='final_angle=-(180-final_angle)'
                final_angle_calc=f'final_angle=-(180-{int(final_angle)})'

            else:
                final_angle=-final_angle
                final_angle_code='final_angle=-final_angle'
                final_angle_calc=f'final_angle=-{int(final_angle)}'
        if type(self.previous_angles)==dict:
           
            if previous_angle and final_angle:

                if (previous_angle<40) and overhook_==1:
                    if final_angle>90:
                        final_angle=-(180-final_angle)
                        final_angle_code='final_angle=-(180-final_angle)'
                        final_angle_calc=f'final_angle=-(180-{int(final_angle)})'

                    else:
                        final_angle_code='final_angle=-final_angle'
                        final_angle_calc=f'final_angle=-{int(final_angle)}'
                        final_angle=-final_angle

        if type(self.previous_angles)==dict:
            if final_angle and previous_angle:
                if abs(final_angle-previous_angle)>30 and final_angle<0:
                    final_angle=0


        if type(self.previous_angles)==dict:
            if final_angle and previous_angle and previous_angle_max:

                if previous_angle_max>10 and final_angle<0:
                    final_angle=0


        #Dictionary containing al the data needed to debug the apical hook script 
        debug_dict={"angle_c":int(angle_c), "angle_s":int(angle_s), "final_angle":int(final_angle),"final_angle_code":final_angle_code, "final_angle_calc":final_angle_calc,"dir_c":int(dir_c),"final_angle":int(final_angle)}
        #  's_top':int(s_top),'s_mid':int(s_mid), 'c_points_upper_x': int(c_points_upper[0]),'c_points_upper_y': int(c_points_upper[1]), 'c_points_bottom_x': int(c_points_bottom[0]),'c_points_bottom_y': int(c_points_bottom[1])}

        return final_angle,debug_dict


    """
        _rectangle
        This function generates points area around cotyledons
        takes x,y points as input and returns and returns the boundaries x1, x2, y1, y2
    """
    def _rectangle(self, x2, y2, radius):
        match=[]
        for row1 in range((x2-radius),(x2+radius)):
            for column1 in range((y2-radius),(y2+radius)):
                match.append((row1, column1))
        return (match)

    def output(self):
        return self._list_array_hypo

