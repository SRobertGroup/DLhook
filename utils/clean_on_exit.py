import os

#Removes data from temporary directory on exit from the seedling software GUI
class remove_data:
    def __init__(self):
        paths=['data/images/','data/predict/','data/postprocess/','data/final_prediction/']#,'post_main/postprocessing_1/','data/final_prediction/']

        for folder in paths:
            files=os.listdir(folder)
            if len(files)>0:
                for file in files:
                    os.remove(os.path.join(folder,file))

if __name__=='__main__':
    remove=remove_data()