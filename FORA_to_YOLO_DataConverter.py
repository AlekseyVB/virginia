import PySimpleGUI as sg
import pandas as pd
import os

type_list = (('Объекты данных FORA','FORA_ObjectsData*.pickle'),)
CLASS_NUM = 2
OUT_DIR = ''

Path = sg.popup_get_file('Открыть файл', no_window=True, file_types=type_list)
if len(Path) != 0: #Проверка что файл был выбран, а не просто закрыто окно
    Directory, FileName = os.path.split(Path)
    ObjectsData = pd.read_pickle(Path)
    sub_out = FileName.replace('FORA_ObjectsData_','').replace('.pickle','')
    out_path = os.path.join(Directory,OUT_DIR+sub_out)
    if not os.path.isdir(out_path):
       os.mkdir(out_path)
    for i in range(2,ObjectsData.shape[0]):
        file_name = ObjectsData.iloc[i,0][:-4]+'.txt'
        width =  ObjectsData.iloc[i,1]
        heigth =  ObjectsData.iloc[i,2]
        coords = ObjectsData.iloc[i,3]
        YoloCoords =[(coords[2]+coords[0])/2/width,
                     (coords[3]+coords[1])/2/heigth,
                     (coords[2]-coords[0])/width,
                     (coords[3]-coords[1])/heigth
                    ]      
        with open(os.path.join(out_path,file_name), "w") as text_file:
            text_file.write('{0}'.format(CLASS_NUM))
            for j in range(len(YoloCoords)):
                text_file.write(' {0:.6f}'.format(YoloCoords[j]))    
else:
    sg.popup('Файл с данными выбран не корректно!')

