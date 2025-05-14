from subprocess import list2cmdline
from tkinter.ttk import Progressbar
from models.unet_model import *
from numpy.lib.function_base import select
from utils.calculate_apical_hook_angle import *
from utils.clean_on_exit import *


from datetime import datetime
import threading
import json
from utils.matching_crop2points_GUI import *
from utils.preprocess_model_input import *
from utils.postprocess_model_output import *
from tkinter.filedialog import asksaveasfilename
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
import ast
from PIL import Image, ImageTk
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import cv2
import tkinter as tk
import pandas as pd
import shutil
import atexit
import sys




#Superres libraries 
from models.superres.superresolution_predict import RealesrganSuperresolution

"""
This is the graphical user interface for the X software



"""
class Gui():
    def __init__(self, root):

       
        self.model_superres=RealesrganSuperresolution()
        sys.setrecursionlimit(10000)


        self.path_crop='data/images/'
        self.path_save_x='data/images/'

        #self.path_crop='E:/Project_seedlings/cropped_images/'
        self.angles=[]
        
        #List of seedling starting points
        self.selected_points=[]

        self.remove_regions=[]

        self.crop_range_points=[]

        self.img_count=0
        self.photo_list=[]

        self.crop_mid_points=[]

        self.transformed_mid_points=[]

        self.activ_button=False


        #The following varoables are used to controll the buttons, 
        self.check_rect1=False
        self.mouse_check=False
        self.check_place_rect=False
        self.crop_window_check=False
        self.rectangle_circle_match=[]
        self.manual_angle=False
        self.point_numb=1
        self.button_circle_check=False
        self.show_image_check=False
        self.button_rect_check=False
        self.rectangle_width=120
        self.rectangle_height=120

        #Boundry points in selected area to be cropped
        self.rectangle_points=[]

        #List with points selected on image
        self.selected_points=[]
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        self.x_length=1100
        self.y_length=650
        width=self.x_length
        height=self.y_length
        
        self.filenames_listbox=False

        self.root=root
        self.root.title('DLhook')
        self.root.geometry(str(self.x_length) + "x" + str(self.y_length))
        self.fin = Frame(self.root, width=200, height=200)
        self.fin.pack()
        self.fin.place(x=0, y=0)
        self.canvas = Canvas(self.fin, bg='#FFFFFF', width=width, height=height)
        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)
        self.btn_import_image = tk.Button(self.root, text="Open image directory", width=17, command=self.add_files)

        self.btn_import_image.place(x=width + 25, y=height - 640)
        
        self.listbox = Listbox(self.root, width=25)
        self.listbox.place(x=width + 10, y=height - 610)

        self.btn_show_image = tk.Button(self.root, text="Show image", width=10, command=self.show_image)
        self.btn_show_image["state"]=DISABLED
        self.btn_show_image.place(x=width + 50, y=height - 440)


        self.btn_sort = tk.Button(self.root, text="Sort images", width=10, command=self.imagename_in_sort_section)
        self.btn_sort["state"]=DISABLED
        self.btn_sort.place(x=width + 50, y=height - 400)
        self.sort_txt_box = tk.Text(self.root,
                   height = 1,
                   width = 40)
        self.sort_txt_box.place(x=width + 10, y=height - 370)
        
        self.sort_start_b=tk.Button(self.root, text="start-string->|", width=10, command=self.start_sort)
        self.sort_start_b["state"]=DISABLED
        self.sort_end_b=tk.Button(self.root, text="|<-end-string", width=10, command=self.end_sort)
        self.sort_end_b["state"]=DISABLED
        self.sort_start_b.place(x=width + 14, y=height - 340)
        self.sort_end_b.place(x=width + 90, y=height - 340)




        self.label_user_input = Label(self.root, 
                  text = "User Input")
        self.label_user_input.place(x=width + 59, y=height - 305)

        self.label_1 = Label(self.root, 
                  text = "1.")
        self.label_1.place(x=width + 20, y=height - 277)
        self.button_place_point=tk.Button(self.root, text="Place points", width=10, command=self.canvas_circle_activate)
        self.button_place_point.place(x=width + 50, y=height - 280)
        self.button_place_point["state"]=DISABLED

        self.label_2 = Label(self.root, 
                  text = "2.")
        self.label_2.place(x=width + 20, y=height - 227)
        self.button_crop = tk.Button(self.root, text="Crop image", width=10, command=self.crop_image)
        self.button_crop.place(x=width + 50, y=height - 230)
        self.button_crop["state"]=DISABLED


        self.label_3 = Label(self.root, 
                  text = "3.")
        self.label_3.place(x=width + 20, y=height - 177)
        self.button_start_analysis = tk.Button(self.root, text="Start Analysis", width=10, command=self._threading_analysis)
        self.button_start_analysis.place(x=width + 50, y=height - 180)
        self.button_start_analysis["state"]=DISABLED





        self.progress = Progressbar(self.root, orient=HORIZONTAL, length=700)
        self.progress.place(x=width - 900, y=height + 15)
        self.progress_bar_label = Label(self.root, 
                  text = "Measurement progress:")
        self.progress_bar_label.place(x=width - 1030, y=height + 15)


        self.debug_var=IntVar(value=1)
        self.check_button_debug= tk.Checkbutton(self.root, text='Debug Mode',variable=self.debug_var)
        self.check_button_debug.place(x=width + 45, y=height + 18)
        #self.button_place_point_remove=tk.Button(self.root, text="Points remove section", width=20, command=self.replace_angle)
        #self.button_place_point_remove.place(x=width + 10, y=height - 260)


        self.angle_p=[]
        self.angle_lines=[]
        self.new_angle_points=[]
        self.line1=False
        self.vinkel=[]


    def start_sort(self):
        self.str1=len(self.sort_txt_box.selection_get())
    def end_sort(self):
        str2=len(self.sort_txt_box.selection_get())
        try:
            if self.str1==len(self.selected_image):
                str1=0
            else:
                str1=self.str1
        except:
            str1=0
        sorted_filenames=[]
        sorted_filenames_output=[]
        for file in self.file_list:
            sorted_filenames.append((int(file[str1:-str2]), file))

        sorted_filenames.sort(key = lambda x: x[0])
        for file1 in sorted_filenames:

            sorted_filenames_output.append(file1[1])


        self.add_files(sorted_filenames_output)
        
    def imagename_in_sort_section(self):
        self.sort_start_b["state"]=NORMAL
        self.sort_end_b["state"]=NORMAL

        clicked_file= self.listbox.curselection()
        for item in clicked_file:
            self.selected_image=self.listbox.get(item)
            self.sort_txt_box.insert(INSERT,self.listbox.get(item))



    def replace_angle(self):

        if self.activ_button==True:
            self.activate_manual_ang()

        file_name=(self.images_final[self.n_angle_image])
        org_filename=self.convert_filename(file_name)

        clicked_file= self.listbox.curselection()
        for item in clicked_file:
            


            data=self.listbox.get(item)
            data=data.split(' ')
            id_n=data[0]
            angle=data[-1]

            list_fill_angles=[]
            for element in self.angle_n:
                if int(element[0]) !=int(id_n):
                    list_fill_angles.append(element)
            

            
            row_id=int(self.new_df[self.new_df['img_name']==org_filename].index.values[0])
            self.new_df.at[row_id, int(id_n)]=self.new_angle


            list_fill_angles.append((int(id_n), self.new_angle))

            list_fill_angles=sorted(list_fill_angles, key=lambda x: x[0])
            self.angle_n=list_fill_angles

            self.listbox.delete(0, END)

            for ang_n in list_fill_angles:
                str1=f'{ang_n[0]}                              {ang_n[1]}°'
                self.listbox.insert(END, str1)
    def replace_angle2(self,Inp=None):
            if self.activ_button==True:
                self.activate_manual_ang()

            file_name=(self.images_final[self.n_angle_image])
            org_filename=self.convert_filename(file_name)

            clicked_file= self.listbox.curselection()
            for item in clicked_file:
                

  

                data=self.listbox.get(item)
                data=data.split(' ')
                id_n=data[0]
                angle=data[-1]

                if Inp:
                    id_n=Inp[0][0]
                    new_angle=Inp[0][1]

                    start_row=Inp[1][0]
                    end_row=Inp[1][1]

                list_fill_angles=[]
                for element in self.angle_n:
                    if int(element[0]) !=int(id_n):
                        list_fill_angles.append(element)
                


                
                row_id=int(self.new_df[self.new_df['img_name']==org_filename].index.values[0])

                for row_id_n in range(start_row,end_row+1):

                    self.new_df.at[row_id_n, int(id_n)]=new_angle

                list_fill_angles.append((int(id_n), new_angle))

                list_fill_angles=sorted(list_fill_angles, key=lambda x: x[0])
                self.angle_n=list_fill_angles

                self.listbox.delete(0, END)

                for ang_n in list_fill_angles:
                    str1=f'{ang_n[0]}                              {ang_n[1]}°'
                    self.listbox.insert(END, str1)



    def place_angle_point(self, event):
        width=200
        height=200
        x1, y1 = event.x, event.y
        r=1

        self.root.bind("<Motion>", self.place_angle_line)
        if (x1 < self.x_length) and (y1 < self.y_length) and self.manual_angle==True:
            if self.check_place_angele == 3:
                self.angle_label.destroy()
                for n in self.angle_p:
                    self.canvas.delete(n)
                self.angle_p=[]

                for m in self.angle_lines:
                    self.canvas.delete(m)
                self.angle_lines = []
                self.check_place_angele = 0
                self.new_angle_points=[]
                self.vinkel=[]

            if self.check_place_angele == 2:
                (x0, y0) = self.points
                self.check_place_angele = self.check_place_angele + 1
                point3 = self.canvas.create_rectangle(x1 - r, y1 - r, x1 + r, y1 + r, stipple='', fill="#b87c84")
                line3 = self.canvas.create_line(x0, y0, x1, y1,fill="#b87c84")



                self.angle_lines.append(line3)

                self.angle_p.append(point3)
                self.points = (x1, y1)
                self.new_angle_points.append((x1, y1))
                self.c = np.array([x1, y1])

            if self.check_place_angele==1:

                (x0,y0)=self.points
                self.check_place_angele = self.check_place_angele + 1
                point2 = self.canvas.create_rectangle(x1 - r, y1 - r, x1 + r, y1 + r, stipple='', fill="green")
                line2 = self.canvas.create_line(x0,y0,x1,y1,fill="#b87c84")

                self.b=np.array([x1,y1])

                self.angle_lines.append(line2)
                self.angle_p.append(point2)
                self.points = (x1, y1)
                self.new_angle_points.append((x1, y1))

            if self.check_place_angele==0:

                self.check_place_angele=self.check_place_angele+1
                point1=self.canvas.create_rectangle(x1 - r, y1 - r, x1 + r, y1 + r, stipple='', fill="green")
                self.angle_p.append(point1)
                self.points = (x1, y1)
                self.new_angle_points.append((x1,y1))
                self.a = np.array([x1, y1])

            if self.check_place_angele == 3:

                ba=self.a - self.b
                bc=self.c - self.b
                cosine_ang=np.dot(ba, bc) /(np.linalg.norm(ba) * np.linalg.norm(bc))
                angle= np.arccos(cosine_ang)
                self.new_angle=round(np.degrees(angle))

                overhook=self.overhook_var.get()
                if overhook==1:
                    self.new_angle=-self.new_angle
                    self.angle_label=tk.Label(root, text =str(self.new_angle)+'°')
                    self.angle_label.place(x=width + 545, y=height + 115)
                else:
                    self.angle_label=tk.Label(root, text =str(self.new_angle)+'°'+' ('+str(180-self.new_angle)+'°)')
                    self.angle_label.place(x=width + 545, y=height + 115)

    def del_angle_line(self):
        self.canvas.delete(self.line1)

    def place_angle_line(self,event):
        if self.check_place_angele<3 and self.manual_angle==True:
            if self.line1!=False:
                self.del_angle_line()

            x1, y1 = event.x, event.y
            (x0, y0)=self.points
            self.line1=self.canvas.create_line(x0, y0, x1 ,y1, fill="#b87c84")

    def rotate_angle_start(self):
        file_name=(self.images_final[self.n_angle_image])
        org_filename=self.convert_filename(file_name)

        clicked_file= self.listbox.curselection()
        for item in clicked_file:

            data=self.listbox.get(item)
            data=data.split(' ')

            id_n=data[0]
            angle=data[-1][:-1]

            self.rotate_data.update({str(id_n):angle})
            
            row_id=int(self.new_df[self.new_df['img_name']==org_filename].index.values[0])
            self.rotate_data.update({str(id_n)+'-start_row':row_id})

    def rotate_angle_end(self):
        file_name=(self.images_final[self.n_angle_image])
        org_filename=self.convert_filename(file_name)

        clicked_file= self.listbox.curselection()
        for item in clicked_file:
            data=self.listbox.get(item)
            data=data.split(' ')
            id_n=data[0]

            #Check if start value for the specific seedling id exists then it will get replaced 
            angle_value=int(self.rotate_data.get(str(id_n)))

            if angle_value:
                end_row=int(self.new_df[self.new_df['img_name']==org_filename].index.values[0])
                start_row=self.rotate_data.get(str(id_n)+'-start_row')
                self.replace_angle2([[int(id_n), angle_value],[start_row,end_row]])
                
                

    def activate_manual_ang(self):
        if self.activ_button==True:
            self.buttonPlace_angle.configure(bg='white')
            for m in self.angle_lines:
                self.canvas.delete(m)



        if self.activ_button==False:
            self.buttonPlace_angle.configure(bg="#d78a5e")
            self.activ_button=True



        if self.manual_angle==False:
            self.angle_points=[]
            self.check_place_angele=0
            self.manual_angle=True
            self.root.bind("<Button-1>", self.place_angle_point)
        else:
            self.manual_angle=False
            for n in self.angle_p:
                self.canvas.delete(n)
            self.angle_p = []

            for m in self.angle_lines:
                self.canvas.delete(m)

    def canvas_circle_activate(self):

        if self.button_circle_check == False:
            self.button_circle_check = True
            self.button_place_point.configure(bg="#d78a5e")

            self.root.bind('<Button-1>', self.place_circle)

        else:
            self.button_crop["state"]=NORMAL
            self.button_circle_check = False
            self.button_place_point.configure(bg="white")
            self.canvas.delete(self.added_oval)
            
            

            if self.debug_var.get() ==0:
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
            #save list with input points from user for faster debug while reruning application
                with open(f"data/debug_data/output/input_starting_points_saved{dt_string}.txt", "w") as output:
                    output.write(str(self.selected_points))


    def place_circle(self, event):
        if (self.button_circle_check==True) and (self.show_image_check==True):
            x1, y1 = event.x, event.y
            r = 4
            if (x1 < self.x_length) and (y1 < self.y_length):

                self.added_oval=self.canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r,stipple='',fill="#536791")

                #self.selected_points.append((x1, y1))
                x_mult = event.x/1100
                y_mult =event.y/650
                x=round(self.width1*x_mult)
                y=round(self.height1*y_mult)
                self.selected_points.append((x, y))

                self.root.mainloop()


    def restart_program(self):
        shutil.rmtree(self.path, ignore_errors=True)
        for file in os.listdir('model_data/images/'):
            os.remove('model_data/images/')
        python = sys.executable
        os.execl(python, python, * sys.argv)
        
    def rect_pos(self, event):
        if self.check_place_rect == True:
            self.r_x, self.r_y = event.x, event.y
            self.crop_mid_points.append((event.x, event.y))
            x_mult = event.x/1100
            y_mult =event.y/650
            x=round(self.width1*x_mult)
            y=round(self.height1*y_mult)
            self.transformed_mid_points.append((x, y))

            if (self.r_x > self.rectangle_width) and (self.r_x < (self.x_length - self.rectangle_width)) and (self.r_y > self.rectangle_height) and (self.r_y < (self.y_length) - self.rectangle_height):
                self.rectangle_points.append((self.r_x- (self.rectangle_width-1), self.r_y -(self.rectangle_height-1), self.r_x+(self.rectangle_width-1), self.r_y+(self.rectangle_height-1)))
                self.canvas.create_rectangle(self.r_x- (self.rectangle_width-1), self.r_y -(self.rectangle_height-1), self.r_x+(self.rectangle_width-1), self.r_y+(self.rectangle_height-1), width=4,outline="#d57239")

    #draws a square on canvas
    def draw_square(self):
        if self.check_place_rect==True:
            if (self.r_x > self.rectangle_width) and (self.r_x < (self.x_length - self.rectangle_width)) and (self.r_y > self.rectangle_height) and (self.r_y < (self.y_length) - self.rectangle_height):
                self.c1=self.canvas.create_rectangle(self.r_x- (self.rectangle_width-1), self.r_y + - (self.rectangle_height-1), self.r_x+(self.rectangle_width-1), self.r_y+(self.rectangle_height-1),width=4,outline="#d78a5e")
                self.check_rect1=True

    #deletes the square from the canvas
    def del_square(self):
        if self.check_rect1==True:
            self.canvas.delete(self.c1)

    #get mouse position for square drawing
    def move(self, event):
        self.del_square()
        self.r_x , self.r_y = event.x, event.y

        self.draw_square()


    #Transform the angle to the correct format
    def _correct_angle(self, angle):
        angle=round(angle)
        if angle>90:
            angle=angle-90
        else:
            angle=angle+90
        return angle


    #function for multithreading analysis in background
    def _threading_analysis(self):
        th = threading.Thread(target=self.start_analysis)
        th.start()

    def _reset_main_window(self):
        a=1


    """
    Starts the analysis of the kinematics batch
    """
    def start_analysis(self):
        #Disable all buttons while analysis is ongoing
        self.button_crop["state"]=DISABLED
        self.button_start_analysis["state"]=DISABLED
        self.button_place_point["state"]=DISABLED
        self.sort_end_b["state"]=DISABLED
        self.sort_start_b["state"]=DISABLED
        self.btn_show_image["state"]=DISABLED
        self.btn_import_image["state"]=DISABLED
        self.btn_sort["state"]=DISABLED


        #
        if self.debug_var.get() ==1:
            try:
                txt_input_files=os.listdir('data/debug_data/input')
                if len(txt_input_files)!=0:
                    with open(os.path.join('data/debug_data/input',txt_input_files[0])) as input_points_txt:
                        input_points_txt=input_points_txt.readlines()
                        self.selected_points=ast.literal_eval(input_points_txt[0])
            except Exception:
                print("no file found!")


        self.crop_mid_points = self.crop_mid_points[:-1] 

        self.selected_points= self.selected_points[:-1]
        self.check_place_rect=False
        self.button_crop.configure(bg="white")


        self.transformed_mid_points=self.transformed_mid_points[:-1]
        match_p=match_crop_points(self.transformed_mid_points, self.selected_points)
        crop_points_distributed=match_p.return_crop_points()




        x1=self.rectangle_points[0][0]
        y1=self.rectangle_points[0][1]
        x2=self.rectangle_points[0][2]
        y2=self.rectangle_points[0][3]
        
        #Initializing preprocess class
        preprocess_init=preprocess_images()



        #Sorting points from left to right in every cropped image
        crop_points_distributed2=[]
        for points_x in crop_points_distributed:
            sorted_point_list=sorted(points_x, key=lambda x: x[0])
            crop_points_distributed2.append(sorted_point_list)
        crop_points_distributed=crop_points_distributed2


        get_point_numbering=point_num(crop_points_distributed)

        temp_get_point_numbering=crop_points_distributed
        for i,list_points in enumerate(temp_get_point_numbering):
            if len(list_points)==0:
                crop_points_distributed.pop(i)
                self.transformed_mid_points.pop(i)



        crop_points_numberID=get_point_numbering.point_numbering()
        #30% of progress bar
        n_images=len(self.file_list)

        for image_n,image in enumerate(self.file_list):
            img_x=cv2.imread(self.path+"/"+image)

            bar_precentage=((image_n/n_images)*30)
            self.progress['value']=bar_precentage
            self.root.update_idletasks()

            #Preprocess Filters, Contrast 
            img_x=preprocess_init.preprocess(img_x)

            for n, points in enumerate(self.transformed_mid_points):
                x_1=points[0]-512
                x_2=points[0]+512
                y_1=points[1]-512
                y_2=points[1]+512

                crop=img_x[y_1:y_2, x_1:x_2]
                if crop.shape == (1024, 1024):
                    #output of superres model is 4x the original image 
                    crop_x4=self.model_superres.enhance(crop)

                    #downsize the image back to its original form
                    crop=cv2.resize(crop_x4, (1024,1024), interpolation = cv2.INTER_AREA)
                    #Superresolution of image

                    cv2.imwrite(self.path_save_x+f'/{n}-crop-{image[:-4]}.png', crop)
        


       

        #deep learning model prediction
        unet_prediction()
        postprocess_masks(crop_points_distributed)
        # try:
        #     postprocess_masks(crop_points_distributed)
        # except:
        #     self.progress['value']=0
        #     return

        

        total_angle=[]
        total_point_num=[]
        filename=[]
        columns_img_angl=['filename','seedling_id','angles','tot_numb']
        img_angle_data=pd.DataFrame(columns=columns_img_angl)
        #img_angle_data=pd.read_csv('img_angle_data.csv')

        #Generate a dictionary with all of the angles for each image crop
        image_crops=[]
        image_crops_angles=[]
        image_crops_angles_max=[]



        for cropped_image_filename in os.listdir('data/images/'):
            if cropped_image_filename[0] not in image_crops:
                image_crops.append(cropped_image_filename[0])
                image_crops_angles.append('.')
                image_crops_angles_max.append('.')

        self.cropped_sorted_filenames=[]
        #check_here
        
        for crop_n in range(len(self.transformed_mid_points)):
            for sorted_filename in self.file_list:
                filename_n=str(crop_n)+'-crop-'+sorted_filename[:-4]+'.png'
                self.cropped_sorted_filenames.append(filename_n)
        
        debug_data_batch={}

        #70% of progress bar
        bar_max=len(self.cropped_sorted_filenames)

        for crop_n,cropped_image_filename in enumerate(self.cropped_sorted_filenames):
            
            bar_precentage=30+int((crop_n/bar_max)*70)
            self.progress['value']=bar_precentage
            self.root.update_idletasks()


            file_name_x=cropped_image_filename

            angles=[]
            point_num1=[]

            crop_id=int(cropped_image_filename[0])


            a1=crop_points_distributed[crop_id]
            a2= crop_points_numberID[crop_id]
            a3=image_crops_angles[int(file_name_x[0])]
            a4=image_crops_angles_max[int(file_name_x[0])]


            init_calculate_angles=apical_hook(cropped_image_filename, crop_points_distributed[crop_id], crop_points_numberID[crop_id],image_crops_angles[int(file_name_x[0])], image_crops_angles_max[int(file_name_x[0])])
            angle_data_x, seedling_id_x ,debug_data=init_calculate_angles.angles()
            debug_data_batch[str(file_name_x)]=debug_data


            dict_angles=dict(zip(seedling_id_x, angle_data_x))
            image_crops_angles[int(file_name_x[0])]=dict_angles

            if image_crops_angles_max[int(file_name_x[0])]=='.':
                image_crops_angles_max[int(file_name_x[0])]=dict_angles
            else:
                angle_keys=list(dict_angles.keys())
                for key in angle_keys:
                    
                    current_value=dict_angles.get(key)

                    previous_max=image_crops_angles_max[int(file_name_x[0])].get(key)
                    if previous_max!=None:
                        if current_value>previous_max:
                            new_value={key:current_value}
                            image_crops_angles_max[int(file_name_x[0])].update(new_value)

        
            data=[[file_name_x, seedling_id_x, angle_data_x, crop_points_numberID[crop_id]]]
            new_df=pd.DataFrame(data, columns=columns_img_angl)
            img_angle_data=pd.concat([img_angle_data, new_df], axis=0)

        img_angle_data.to_csv('img_angle_data.csv', index=False)
        # json_debug_data=json.dump(debug_data_batch)
        if self.debug_var.get() ==0:
            with open('data/json_data/json_data.json', 'w') as outfile:
                json.dump(debug_data_batch, outfile)

        self.save_data()
        self.img_angle_data=img_angle_data
        self.openNewWindow()




    """
    Crop image into patches with the region of interest containing the seedlings to be analyzed
    """
    def crop_image(self):

        if self.check_place_rect==False:
            self.button_crop.configure(bg="#d78a5e")
            self.root.bind("<Motion>",self.move)
            self.root.bind("<Button-1>", self.rect_pos)
            self.mouse_check=True
            self.check_place_rect=True

        else:
            self.button_crop.configure(bg="white")
            self.root.bind("<Motion>",self.move)
            self.root.bind("<Button-1>", self.rect_pos)
            self.mouse_check=False
            self.check_place_rect=False
            self.button_start_analysis["state"]=NORMAL
       

    #Converts the crop filename to the original filename
    def convert_filename(self, filename):
        filename_n=filename[7:-4]+self.img_format
        return(filename_n)



    def next_angle_img(self):
        width=200
        height=200
        self.listbox.delete(0, END)

        if self.n_angle_image!=self.checker_next_img_limit-1:
            self.n_angle_image=self.n_angle_image+1

            file_name=(self.images_final[self.n_angle_image])
            self.current_img_name.destroy()
            self.current_img_name=tk.Label(root,text =file_name[:-4])
            self.current_img_name.place(x=width +500,y=height - 160)
            angle_dataframe_x=self.angle_dataframe.loc[self.angle_dataframe['filename'] == str(file_name)]


            org_filename=self.convert_filename(file_name)
            tot_numb=ast.literal_eval(angle_dataframe_x['tot_numb'].tolist()[0])
            angle_data=self.new_df.loc[self.new_df['img_name']==org_filename]
            angle_d = angle_data.squeeze()
            
            self.angle_n=[]
            for seedling_id_n in tot_numb:
                angle=(angle_d[int(seedling_id_n)])

                if np.isnan(angle):
                    self.angle_n.append((seedling_id_n,'-'))
                else:
                    self.angle_n.append((seedling_id_n,int(angle)))

            for ang_n in self.angle_n:
                str1=f'{ang_n[0]}                              {ang_n[1]}°'
                self.listbox.insert(END, str1)
            img_1=cv2.imread(self.img_final_folder+self.images_final[self.n_angle_image],1)
            img_1=cv2.resize(img_1, (600,600))
            photo_n = ImageTk.PhotoImage(image=Image.fromarray(img_1))
            self.canvas.delete(self.image_on_canvas)
            self.image_on_canvas = self.canvas.create_image(0, 0, image=photo_n, anchor=tk.NW)        
            self.root.mainloop()

        else:
            self.save_button['state']=NORMAL



    def previous_angle_img(self):
            width=200
            height=200
            self.listbox.delete(0, END)
            if self.n_angle_image!=0:
                self.n_angle_image=self.n_angle_image-1
            if self.n_angle_image<self.checker_next_img_limit:
                file_name=(self.images_final[self.n_angle_image])
                self.current_img_name.destroy()
                self.current_img_name=tk.Label(root,text =file_name[:-4])
                self.current_img_name.place(x=width +500,y=height - 160)
                angle_dataframe_x=self.angle_dataframe.loc[self.angle_dataframe['filename'] == str(file_name)]


                org_filename=self.convert_filename(file_name)
                tot_numb=ast.literal_eval(angle_dataframe_x['tot_numb'].tolist()[0])
                angle_data=self.new_df.loc[self.new_df['img_name']==org_filename]
                angle_d = angle_data.squeeze()
                
                self.angle_n=[]
                for seedling_id_n in tot_numb:
                    angle=(angle_d[int(seedling_id_n)])

                    if np.isnan(angle):
                        self.angle_n.append((seedling_id_n,'-'))
                    else:
                        self.angle_n.append((seedling_id_n,int(angle)))

                for ang_n in self.angle_n:
                    str1=f'{ang_n[0]}                              {ang_n[1]}°'
                    self.listbox.insert(END, str1)
                img_1=cv2.imread(self.img_final_folder+self.images_final[self.n_angle_image],1)
                img_1=cv2.resize(img_1, (600,600))
                photo_n = ImageTk.PhotoImage(image=Image.fromarray(img_1))
                self.canvas.delete(self.image_on_canvas)
                self.image_on_canvas = self.canvas.create_image(0, 0, image=photo_n, anchor=tk.NW)        
                self.root.mainloop()

    def openNewWindow(self):
        #remove buttons
        self.n_angle_image=0
        self.listbox.destroy()
        self.btn_import_image.destroy()
        self.btn_show_image.destroy()
        self.button_crop.destroy()
        self.button_place_point.destroy()
        self.btn_sort.destroy()
        self.sort_txt_box.destroy()
        self.sort_start_b.destroy()
        self.sort_end_b.destroy()
        self.label_user_input.destroy()
        self.label_1.destroy()
        self.label_2.destroy()
        self.label_3.destroy()
        self.progress.destroy()
        self.button_start_analysis.destroy()
        self.check_button_debug.destroy()
        self.progress_bar_label.destroy()


        


        

        self.fin.destroy()

        width=200
        height=200
        self.fin = Frame(self.root, width=200, height=200)
        self.fin.pack()
        self.fin.place(x=20, y=20)
        self.canvas = Canvas(self.fin, bg='#FFFFFF', width=600, height=600)
        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)


        label_rotate=tk.Label(root, text ='Cotyledon rotation')
        label_rotate.place(x=width + 463, y=height + 256)

        
        self.button_rot_start = tk.Button(self.root, text="Start rot", width=6, command=self.rotate_angle_start)
        self.button_rot_start.place(x=width + 464, y=height + 281)


        self.button_rot_end = tk.Button(self.root, text="End rot", width=6, command=self.rotate_angle_end)
        self.button_rot_end.place(x=width + 518, y=height + 281)

        self.rotate_data={}


        label_show_image=tk.Label(root, text ='Show prediction')
        label_show_image.place(x=width + 473, y=height + 330)
        

        self.buttonNext_angle_img = tk.Button(self.root, text="→", width=4, command=self.next_angle_img)
        self.buttonNext_angle_img.place(x=width + 520, y=height + 355)


        self.buttonPrevious_angle_img = tk.Button(self.root, text="←", width=4, command=self.previous_angle_img)
        self.buttonPrevious_angle_img.place(x=width + 480, y=height + 355)
        
        self.save_button = tk.Button(self.root, text="Save data", width=15, command=self.save_csv_data)
        self.save_button.place(x=width + 460, y=height + 400)
        self.save_button['state']=DISABLED


        label_manual_ang=tk.Label(root, text ='Manual angle :')
        label_manual_ang.place(x=width + 460, y=height + 115)

        self.overhook_var=IntVar()
        self.check_button_overhook= tk.Checkbutton(self.root, text='Overhook',variable=self.overhook_var)
        self.check_button_overhook.place(x=width + 460, y=height + 150)

        self.buttonPlace_angle = tk.Button(self.root, text="Place angle", width=15, command=self.activate_manual_ang)
        self.buttonPlace_angle.place(x=width + 460, y=height + 180)
        self.buttonReplace_angle = tk.Button(self.root, text="Replace angle", width=15, command=self.replace_angle)
        self.buttonReplace_angle.place(x=width + 460, y=height + 210)


        Label_img_name = tk.Label(root,text ='Image :')
        Label_img_name.place(x=width +450, y=height - 160)

        Label_seedling = tk.Label(root,text ='#Seedling')
        Label_seedling.place(x=width + 450, y=height - 120)
        Label_seedling_angle = tk.Label(root,text ='Angle')
        Label_seedling_angle.place(x=width + 550, y=height - 120)


        self.listbox = Listbox(self.root, width=22, height=12)
        self.listbox.place(x=width + 450, y=height - 100)
        self.img_final_folder='data/final_prediction/'
        self.images_final=self.cropped_sorted_filenames

        self.checker_next_img_limit=len(self.images_final)


        self.angle_dataframe=pd.read_csv('img_angle_data.csv')

        angle_dataframe_x=self.angle_dataframe.loc[self.angle_dataframe['filename'] == self.images_final[0]]

        self.current_img_name = tk.Label(root,text =self.images_final[0][:-4])
        self.current_img_name.place(x=width +500,y=height - 160)
        self.angle_n=[]
        seed_id=eval(angle_dataframe_x['seedling_id'].tolist()[0])
        angles=eval(angle_dataframe_x['angles'].tolist()[0])
        tot_numb=eval(angle_dataframe_x['tot_numb'].tolist()[0])
        for n_ang, angle_N in enumerate(angles):
            id_x=seed_id[n_ang]
            if id_x==None:
                id_x=1
            self.angle_n.append((id_x, angle_N))
            
        for ang_n in self.angle_n:
            str1=f'{ang_n[0]}                              {ang_n[1]}°'
            self.listbox.insert(END, str1)
        img_1=cv2.imread(self.img_final_folder+self.images_final[0],1)

        img_1=cv2.resize(img_1, (600, 600))
        photo_n = ImageTk.PhotoImage(image=Image.fromarray(img_1))

        self.image_on_canvas = self.canvas.create_image(0, 0, image=photo_n, anchor=tk.NW)
        self.root.geometry("%dx%d+0+0" % (825, 650))
        self.root.mainloop()

    def save_csv_data(self):
        file_formate=[('CSV-file','*.csv')]
        save_file_path = asksaveasfilename(filetypes= file_formate, defaultextension=file_formate)
        self.new_df.to_csv(save_file_path, index=False)
        """
        This function formats the data in the final form and save it to the path that the user gives.
        The data is saved with the filenames(timepoints) as the first  column, the resto of the columns represtnts
        each seedling-id-number 
        """

    def save_data(self):
        df=pd.read_csv('img_angle_data.csv')

        file_names=df['filename'].tolist()

        def list_to_string(s):
            string_n=s.split('-')
            string_n=string_n[2:]

            str1='-'
            return(str1.join(string_n))

        files=self.file_list
        crops=[n for n in range(len(self.transformed_mid_points))]



        img_num=(len(file_names)/len(crops))

        img_name_matrix=[['' for n in range(len(crops))] for i in range(int(img_num))]

        crop_filenames=[n for n in range(len(file_names))]
        for n in range(int(img_num)):
            filename_index=crop_filenames[n::int(img_num)]
            for i,index in enumerate(filename_index):
                img_name_matrix[n][i]=file_names[index]

        new_list=df['tot_numb'].iloc[-1]
        new_list = ast.literal_eval(new_list)
        last_seedling_id=new_list[-1]
        data=['img_name']
        for n in range(1, last_seedling_id+1):
            data.append(n)



        new_list=df['tot_numb'].iloc[-1]
        new_list = ast.literal_eval(new_list)
        last_seedling_id=new_list[-1]
        data=['img_name']
        for n in range(1, last_seedling_id+1):
            data.append(n)


        self.new_df=pd.DataFrame(columns=data)
        main_ang=[]
        main_ids=[]
        for n,element in enumerate(img_name_matrix):
            angle_data=[files[n]]
            seedling_ids_list=['img_name']
            for name in element:
                df_n=df.loc[df['filename']==name]
                angles=df_n['angles'].tolist()
                angles=angles[0]
                angles=ast.literal_eval(angles)
                angle_data=angle_data+angles
                
                seedling_ids=df_n['seedling_id'].tolist()
                seedling_ids=seedling_ids[0]
                seedling_ids=ast.literal_eval(seedling_ids)
                seedling_ids_list=seedling_ids_list+seedling_ids
            main_ang.append(angle_data)
            main_ids.append(seedling_ids_list)
        for n in range(len(img_name_matrix)):
            zip_iterator = zip(main_ids[n], main_ang[n])
            a_dictionary = dict(zip_iterator)
            self.new_df = self.new_df.append(a_dictionary, ignore_index=True)
        




    def add_files(self, path_input=None):
        if path_input!=None:
            self.listbox.delete(0, tk.END)
            for file_n in path_input:
                self.listbox.insert(END, file_n)
            self.file_list=path_input

        if self.filenames_listbox==True and path_input==None:
            self.listbox.delete(0, tk.END)
            self.path=filedialog.askdirectory()
            folder=self.path.split('/')
            self.save_path=self.path_crop+folder[-1]

            self.file_list=sorted(os.listdir(self.path))

            path=sorted(os.listdir(self.path))


            for file_n in path:
                self.listbox.insert(END, file_n)


        if self.filenames_listbox==False and path_input==None:
            #Activate buttons
            self.btn_show_image["state"]=NORMAL
            self.btn_sort["state"]=NORMAL


            self.path=filedialog.askdirectory()
            folder=self.path.split('/')
            self.save_path=self.path_crop+folder[-1]
            self.file_list=os.listdir(self.path)
            path=sorted(os.listdir(self.path))

            for file_n in path:
                self.listbox.insert(END, file_n)

            self.filenames_listbox=True

        
    #Displays the image in the tkinter window 
    def show_image(self):
        self.button_place_point["state"]=NORMAL
        self.sort_start_b["state"]=NORMAL
        self.sort_end_b["state"]=NORMAL

        clicked_file= self.listbox.curselection()
        for item in clicked_file:

            if self.show_image_check==True:
                
                #Image-file format
                self.img_format=self.listbox.get(item)[-4:]

                self.canvas.delete(self.image_on_canvas)

                self.image_n = cv2.imread(self.path+"/"+self.listbox.get(item))
                image_n=self.image_n
                height1, width1, _ = image_n.shape
                self.height1, self.width1, _ = image_n.shape

        
                #Ratio of image axis after reshape
                self.Rx=(1100/((self.rectangle_width*2)-2))
                self.Ry=(650/((self.rectangle_height*2)-2))


                image_n=cv2.resize(image_n, (1100,650))
                self.image_n2=image_n

                photo_n = ImageTk.PhotoImage(image=Image.fromarray(image_n))
                self.image_on_canvas = self.canvas.create_image(0, 0, image=photo_n, anchor=tk.NW)
                self.root.mainloop()


            if self.show_image_check==False:

                self.show_image_check=True

                image_n = cv2.imread(self.path+"/"+self.listbox.get(item))
                self.img_format=self.listbox.get(item)[-4:]
                height1, width1, channels1 = image_n.shape
                self.height1, self.width1, _ = image_n.shape
                
                self.Rx=(1100/((self.rectangle_width*2)-2))
                self.Ry=(650/((self.rectangle_height*2)-2))
                image_n=cv2.resize(image_n, (1100,650))

                photo_n = ImageTk.PhotoImage(image=Image.fromarray(image_n))
                self.image_on_canvas = self.canvas.create_image(0, 0, image=photo_n, anchor=tk.NW)
                self.root.mainloop()


#Remove image files at exit of software
def remove_files_exit():
    remove_data()

atexit.register(remove_files_exit)

if __name__== '__main__':
    remove_files_exit()
    root=tk.Tk()
    gui=Gui(root)
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (1280, 700))
    root.mainloop()