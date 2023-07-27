# Гайд источник: https://youtu.be/2QA36DYxMUw
# https://www.pysimplegui.org/en/latest/
# https://habr.com/ru/companies/vdsina/articles/557316/
# Установка библиотек
# py -3 -m pip install PySimpleGUI
# py -3 -m pip install pillow
# py -3 -m pip install screeninfo
# py -3 -m pip install pandas


import PySimpleGUI as sg
from PIL import Image, ImageGrab
from io import BytesIO
from screeninfo import get_monitors
import pandas as pd
import random
import os

for m in get_monitors():
  if m.is_primary:
    MONITOR_WIDTH = m.width
    MONITOR_HEIGHT = m.height

if MONITOR_WIDTH < 1280 or MONITOR_HEIGHT < 720:
  sg.popup('Требуется минимальное разрешение монитора 1280х720', title = 'Предупреждение')

# Корректировка положения окна
DELTA_X = -8
DELTA_Y = -45
INTERFACE_X = 20
INTERFACE_Y = 180

# Определение параметров canvas-graph
MAX_WIDTH = MONITOR_WIDTH - INTERFACE_X
MAX_HEIGHT = MONITOR_HEIGHT - INTERFACE_Y
dx = 0
dy = 0

# Переменные
type_list = (('Изображения','*.jpg;*.bmp;*.gif;*.png'),)
Marking = False
Current = 0
AutoObjects = True
AutoFrames = True
AutoMove = True
MoveOff = True
FastFrames = 25
FastestFrames = 250
Directory = ''
DEFAULT_FILE_NAME = 'Пустой.jpg'
ObjectsData = pd.DataFrame({'File': ['Настройки',DEFAULT_FILE_NAME],'Width':[3,MAX_WIDTH],'Height':['red',MAX_HEIGHT],'Объект 1':['blue',[0,0,0,0]],'Объект 2':['green',[0,0,0,0]],'Объект 3':['Orange',[0,0,0,0]]})

# Инициализация переменных
FileName = ObjectsData['File'].iloc[1]
img_width = ObjectsData[ObjectsData['File'] == FileName]['Width'].iloc[0]
img_height = ObjectsData[ObjectsData['File'] == FileName]['Height'].iloc[0]
width = MAX_WIDTH
height = MAX_HEIGHT

FileNumber = 1
MaxObjects = ObjectsData.shape[1]-3
line_width = ObjectsData['Width'].iloc[0]

about = '''
Fast Objects Rectangle Annotator (FORA) - программа разметки изображений
посредством Ограничительных Рамок (Bounding Boxes) для рещения задач 
Обнаружения Объектов (Object Detection).

Цель поставленная при создании данной программы:
минимизация действий пользователя по выбору типов объектов
и переключению между изображениями.'''

# Окно и компоненты
def Create_Window():

  global window, graph
  # Извлечение параметров отображения по имени файла
  
  sg.theme('BrownBlue')

  menu_def = [ ['Файл', ['Открыть','Сохранить','Выход']],
               ['Настройки', ['Выбрать цвет перекрестия', 'Увеличить толщину линий разметки', 'Уменьшить толщину линий разметки']],
             ['Информация', 'О программе']
           ]

  layout = [
            [sg.Menu(menu_def,)],
            [sg.Push(),
             sg.Text('Объект '+str(Current+1)+' из', key = 'cur_obj'),
             sg.Input(MaxObjects, key = 'max_obj', size = (2, 1), enable_events= True, justification = 'center', focus = 'False'),          
             sg.Button('\u2bc7', key = 'prev_obj'),
             sg.Button('\u2bc8', key = 'next_obj'),
             sg.Input('  ', key = 'obj_color', size = (2, 1), background_color = ObjectsData.iloc[0,3+Current], focus = 'False'),
             sg.Input(ObjectsData.columns[3+Current], key = 'obj_name', enable_events= True, size = (20, 1), focus = 'False'),
             sg.Button('\u2716', key = 'delete_obj'),
             sg.VerticalSeparator(pad=None),
             sg.Text('Кадры:'),
             sg.Button('\u276e\u276e\u276e', key = 'fastest_prev', enable_events= True),
             sg.Button('\u276e\u276e', key = 'fast_prev', enable_events= True),
             sg.Button('\u276e', key = 'prev_frame', enable_events= True),
             sg.Input(FileNumber, key = 'file_num', enable_events = True, size = (6, 1), focus = 'False', justification = 'center'),
             sg.Button('\u276f', key = 'next_frame', enable_events= True),
             sg.Button('\u276f\u276f', key = 'fast_next', enable_events= True),
             sg.Button('\u276f\u276f\u276f', key = 'fastest_next', enable_events= True),
             sg.VerticalSeparator(pad=None),
             sg.Text('Смещение: dX'),
             sg.Input(str(dx), key = 'delta_x', size = (3, 1), enable_events= True, justification = 'center', focus = 'False'),
             sg.Button('+',size = (1, 1)),
             sg.Button('-',size = (1, 1)),
             sg.Text('dY'),
             sg.Input(str(dy), key = 'delta_y', size = (3, 1), enable_events= True, justification = 'center', focus = 'False'),
             sg.Button('+',size = (1, 1)),
             sg.Button('-',size = (1, 1)),
             sg.VerticalSeparator(pad=None),
             sg.Text('Авто:'),
             sg.Checkbox('Объекты',  key = 'auto_obj', enable_events= True, default = AutoObjects, disabled = False),
             sg.Checkbox('Кадры',  key = 'auto_fra', enable_events= True, default = AutoFrames, disabled = False),
             sg.Checkbox('Смещение',  key = 'auto_mov', enable_events= True, default = AutoMove, disabled = MoveOff),
             sg.Push()
            ],
            [sg.Push(),
             sg.Graph(
              canvas_size = (width, height),
              graph_bottom_left=(0,height),
              graph_top_right=(width, 0),
              background_color="white",
              key="graph"),
             sg.Push()
            ],
            [sg.Text('Имя файла: ' +FileName, key = 'file_nam'),
             sg.Text('Толщина линий разметки: '+str(ObjectsData['Width'].iloc[0]), key = 'line_wth'),
            ]
           ]
  
  window = sg.Window('FORA - Fast Objects Rectangle Annotator', layout, relative_location = (DELTA_X, DELTA_Y), margins =(0, 0), finalize=True)

  # Обработчики событий
  graph = window['graph']
  graph.bind('<Motion>','_mouse_motion') 
  graph.bind('<Button-1>','_mouse_down')
  graph.bind('<B1-Motion>','_mouse_drag')
  graph.bind('<Double-Button-1>','_double_click')
  graph.bind('<ButtonRelease-1>','_mouse_up')
  graph.bind('<ButtonRelease-3>','_right_click')
  graph.bind('<Button-2>','_midmouse_down')
  graph.bind('<ButtonRelease-2>','_midmouse_up')
  graph.bind('<B2-Motion>','_midmouse_drag')
  
  window['obj_color'].bind("<Button-1>", '_select_color')

# Перерисовка окна
def Redraw_Objects():
  global graph
  
  # Перерисовка объектов
  for i in range(3, MaxObjects+3):
    # Рамки рисуются внутрь от сохраненных координат
    rectangle = graph.draw_rectangle((ObjectsData[ObjectsData['File'] == FileName].iloc[0,i][0]+dx,
                                      ObjectsData[ObjectsData['File'] == FileName].iloc[0,i][1]+dy),
                                     (ObjectsData[ObjectsData['File'] == FileName].iloc[0,i][2]+dx-line_width,
                                      ObjectsData[ObjectsData['File'] == FileName].iloc[0,i][3]+dy-line_width),
                                      fill_color = None,
                                      line_color = ObjectsData.iloc[0,i],
                                      line_width = line_width)

def Redraw_Frame():
  
  global width, height, img_width, img_height, height,data, window, graph, dx, dy, MoveOff
  
  # Проверка что изменилось разрешение изображения
  if ObjectsData['Width'].iloc[FileNumber] != img_width or ObjectsData['Height'].iloc[FileNumber] != img_height:
    img_width = ObjectsData['Width'].iloc[FileNumber]
    img_height = ObjectsData['Height'].iloc[FileNumber]
    width = img_width
    height = img_height
    # Корректировка положения рисунка в окне
    if img_width < MAX_WIDTH and img_height < MAX_HEIGHT:
      dx = 0
      dy = 0
      MoveOff = True
    # Корректировка положения рисунка в окне
    if img_width > MAX_WIDTH:
      if AutoMove: dx = (width - MAX_WIDTH)//2
      width = MAX_WIDTH
      MoveOff = False
    if img_height > MAX_HEIGHT:
      if AutoMove: dy = (height - MAX_HEIGHT)//2
      height = MAX_HEIGHT
      MoveOff = False

    im = Image.open(os.path.join(Directory,ObjectsData['File'].iloc[FileNumber]))
    window.close()
    Create_Window()
  else:
    im = Image.open(os.path.join(Directory,ObjectsData['File'].iloc[FileNumber]))
    graph.erase()
    Update_Obj_Info()

  with BytesIO() as output:
    im.save(output, format="PNG")
    data = output.getvalue()
    graph.draw_image(data=data, location=(-dx,-dy))
    Redraw_Objects()

# Обновление информации об объекте
def Update_Obj_Info():
  window['obj_color'].update(background_color = ObjectsData.iloc[0,3+Current])
  window['max_obj'].update(MaxObjects)
  window['obj_name'].update(ObjectsData.columns[3+Current])
  window['cur_obj'].update('Объект '+str(Current+1)+' из')
  window['file_num'].update(FileNumber)
  window['file_nam'].update('Имя файла: ' +FileName)
  

# Функция сохранения
def Save_Canvas(FileName):
  widget = graph.Widget
  box = (widget.winfo_rootx(), widget.winfo_rooty(),
         widget.winfo_rootx() + widget.winfo_width(),
         widget.winfo_rooty() + widget.winfo_height()
        )
  grab = ImageGrab.grab(bbox=box)
  grab.save(FileName)

def Make_Files_List(Directory, FileName):
  FilesList = []
  for file in os.listdir(Directory):
    _, ext = os.path.splitext(file)
    if ext in ['.png', '.bmp', '.jpg', '.gif','.jpeg']:
      FilesList.append(file)
  FilesList.sort()
  return FilesList

def Update_DataFile(ObjectsData, FilesList, Directory, FileName):
  exist_files_counter = 0
  # Удаление записи для пустого файла из ObjectsData по умолчанию
  ObjectsData = ObjectsData[ObjectsData['File'] != 'Пустой']
  # Проход по файлам из списка

  for file in FilesList:
    if ObjectsData[ObjectsData['File'] == file].shape[0] == 0:
      im = Image.open(os.path.join(Directory,file))
      width, height = im.size
      line = [file, width, height]
      for k in range(ObjectsData.shape[1]-3):
        line.append([0,0,0,0])
      ObjectsData.loc[len(ObjectsData.index)] = line
    else:
      exist_files_counter += 1
  # Вывод результатов обработки
  if exist_files_counter > 0:
    sg.popup('Для '+str(exist_files_counter)+' изображений из '+str(len(FilesList))+' найдены ранее сохраненные данные.')
  else:
    sg.popup('Для обработки добавлено '+str(len(FilesList))+' изображений.')

  if ObjectsData.shape[0]-1 > len(FilesList):
    sg.popup('Собранных данных больше чем изображений. Возможно из ранее обработанной папки были удалены файлы!')
  # Определяю индекс открытого файла в итоговой таблице
  FileNumber = ObjectsData[ObjectsData['File'] == FileName].index.values[0]
  return ObjectsData, FileNumber

# Создаю первоначальное окно
Create_Window()

v_line = graph.draw_line((0,0),(0,0), width = 0, color=ObjectsData['Height'].iloc[0])
h_line = graph.draw_line((0,0),(0,0), width = 0, color=ObjectsData['Height'].iloc[0])


# Чтобы в изначальном окне можно было тестировать инструменты создаю пустое полотно для отображения
im = Image.new("RGB", (width, height), (255, 255, 255))
with BytesIO() as output:
  im.save(output, format="PNG")
  data = output.getvalue()

# Сохраняю пустой кадр, чтобы не вылетала программа при переключении кадров без загруженных файлов
Save_Canvas(DEFAULT_FILE_NAME)

# Обработчик событий
while True:
  event, values = window.read()
  #print(f'event:{event} values: {values}')
  
  # Обработка движения мыши
  if event == 'graph_mouse_motion':
    graph.delete_figure(v_line)
    graph.delete_figure(h_line)
    x,y = values['graph']
    v_line = graph.draw_line((x,0),(x,height), width = ObjectsData['Width'].iloc[0], color=ObjectsData['Height'].iloc[0])
    h_line = graph.draw_line((0,y),(width,y), width = ObjectsData['Width'].iloc[0], color=ObjectsData['Height'].iloc[0])

  else:

    if event == sg.WIN_CLOSED or event == 'Выход':
      window.close()
      break
  
    # Обработка меню
    elif event == 'Открыть':
        Path = sg.popup_get_file('Открыть файл', no_window=True, file_types=type_list)
        if len(Path) != 0: #Проверка что файл был выбран, а не просто закрыто окно
          Directory, FileName = os.path.split(Path)
          if os.path.exists(os.path.join(Directory,'FORA_ObjectsData.pickle')):
            ObjectsData = pd.read_pickle(os.path.join(Directory,'FORA_ObjectsData.pickle'))
            sg.popup('Файл FORA_ObjectsData.csv\n с данными об изображениях и объектах найден и загружен из папки:\n' + str(Directory)+'\nЕсли изменялось содержимое папки, то рекомендуется проверить файл через соответствующую опцию в меню.')
          else:
            FilesList = Make_Files_List(Directory, FileName)
            ObjectsData, FileNumber = Update_DataFile(ObjectsData, FilesList, Directory, FileName)
          MaxObjects = ObjectsData.shape[1]-3
          line_width = ObjectsData['Width'].iloc[0]
          Redraw_Frame()

    elif event == 'Сохранить':
      ObjectsData.to_pickle(os.path.join(Directory,'FORA_ObjectsData.pickle'), compression='infer', protocol=5, storage_options=None)
      ObjectsData.to_csv(os.path.join(Directory,'FORA_ObjectsData.csv'))
      sg.popup('Файл FORA_ObjectsData.pickle с данными об изображениях и объектах сохранен в папке: ' + str(Directory))
            
    elif event == 'Выбрать цвет перекрестия':
      new_color = sg.askcolor()[1]
      if new_color:
        ObjectsData['Height'].iloc[0] = new_color
    
    elif event == 'Увеличить толщину линий разметки':
      if ObjectsData.iloc[0,1] < 7:
        ObjectsData.iloc[0,1] += 1
        line_width = ObjectsData.iloc[0,1]
        window['line_wth'].update('Толщина линий разметки: '+str(ObjectsData['Width'].iloc[0]))
        graph.erase()
        graph.draw_image(data=data, location=(-dx,-dy)) 
        Redraw_Objects()        
      else: sg.popup('Достигнут предел толщины линии!')

    elif event == 'Уменьшить толщину линий разметки':
      if ObjectsData.iloc[0,1] > 1:
        ObjectsData.iloc[0,1] -= 1
        line_width = ObjectsData.iloc[0,1]
        window['line_wth'].update('Толщина линий разметки: '+str(ObjectsData['Width'].iloc[0]))
        graph.erase()
        graph.draw_image(data=data, location=(-dx,-dy)) 
        Redraw_Objects()        
      else: sg.popup('Толщина линии минимальна!')
    
    elif event == 'О программе':
      sg.popup('О программе:',about)

    # Обработка полей ввода и кнопок

    elif event == 'obj_color_select_color':
      new_color = sg.askcolor()[1]
      if new_color:
        window['obj_color'].update(background_color = new_color)
        ObjectsData.iloc[0,3+Current] = new_color
      Redraw_Objects()

    elif event == 'max_obj':
      while event == 'max_obj':
        event, values = window.read()
      try:
        NewMaxObjects = int(values["max_obj"])
        if NewMaxObjects > MaxObjects:
          if ObjectsData.shape[1]-3 <= NewMaxObjects:
            # Добавляю новые столбцы
            for i in range(ObjectsData.shape[1]-2, NewMaxObjects+1):
              name = 'Объект '+str(i)
              color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
              column = [color]
              for k in range(ObjectsData.shape[0]-1):
                column.append([0,0,0,0])
              ObjectsData[name] = column

        MaxObjects = NewMaxObjects
        if Current > MaxObjects-1:
          Current = MaxObjects-1
          window['cur_obj'].update('Объект '+str(Current+1)+' из')
      except:
        window['obj_name'].update(MaxObjects)
        sg.popup('Введите целое число!')               
        
    elif event == 'file_num':
      while event == 'file_num':
        event, values = window.read()
      try:
        NewFileNumber = int(values["file_num"])
        if NewFileNumber > ObjectsData.shape[0] - 1:
          sg.popup('В папке всего ' + str(ObjectsData.shape[0] - 1)+' кадра(ов)')
        elif NewFileNumber > 0:
          FileNumber = NewFileNumber
          FileName = ObjectsData['File'].iloc[FileNumber]
          Redraw_Frame()  
        else: sg.popup('Введите целое положительное число от 1 до ' + str(ObjectsData.shape[0] - 1))  
      except:
        window['file_num'].update(FileNumber)
        sg.popup('Введите целое положительное число!')  

    elif event == 'obj_name':
      old_name = ObjectsData.columns[3+Current]
      while event == 'obj_name':
        event, values = window.read()
      try:
        ObjectsData.rename(columns = {old_name : values["obj_name"]}, inplace = True)
      except:
        window['obj_name'].update(old_name)
        sg.popup('Недопустимое имя слоя!')

    elif event == 'prev_obj':
      if Current == 0:
        Current = MaxObjects-1
      else:
        Current -= 1
      Update_Obj_Info()

    elif event in ('next_obj', 'graph_right_click'):
      if Current == MaxObjects-1:
        Current = 0
      else:
        Current += 1
      Update_Obj_Info()

    elif event in ('delete_obj'):
        ObjectsData.iloc[FileNumber,Current+3][0] = 0
        ObjectsData.iloc[FileNumber,Current+3][2] = 0
        ObjectsData.iloc[FileNumber,Current+3][1] = 0
        ObjectsData.iloc[FileNumber,Current+3][3] = 0
        graph.erase()
        graph.draw_image(data=data, location=(-dx,-dy)) 
        Redraw_Objects()

    elif event in ('next_frame', 'graph_double_click'):
      Current = 0
      if FileNumber == ObjectsData.shape[0]-1:
        FileNumber = 1
      else:
        FileNumber += 1
      FileName = ObjectsData['File'].iloc[FileNumber]
      Redraw_Frame()

    elif event in ('fast_next'):
      Current = 0
      if FileNumber + FastFrames <= ObjectsData.shape[0]-1:
        FileNumber += FastFrames
      else:
        FileNumber = ObjectsData.shape[0]-1
      FileName = ObjectsData['File'].iloc[FileNumber]
      Redraw_Frame()

    elif event in ('fastest_next'):
      Current = 0
      if FileNumber + FastestFrames <= ObjectsData.shape[0]-1:
        FileNumber += FastestFrames
      else:
        FileNumber = ObjectsData.shape[0]-1
      FileName = ObjectsData['File'].iloc[FileNumber]
      Redraw_Frame()

    elif event in ('prev_frame'):
      Current = 0
      if FileNumber == 1:        
        FileNumber = ObjectsData.shape[0]-1
      else:
        FileNumber -= 1
      FileName = ObjectsData['File'].iloc[FileNumber]
      Redraw_Frame()

    elif event in ('fast_prev'):
      Current = 0
      if FileNumber - FastFrames <= 1:        
        FileNumber = 1
      else:
        FileNumber -= FastFrames
      FileName = ObjectsData['File'].iloc[FileNumber]
      Redraw_Frame()

    elif event in ('fastest_prev'):
      Current = 0
      if FileNumber - FastestFrames <= 1:        
        FileNumber = 1
      else:
        FileNumber -= FastestFrames
      FileName = ObjectsData['File'].iloc[FileNumber]
      Redraw_Frame()

    elif event == 'auto_obj':
      AutoObjects = values["auto_obj"]
 
    # Обработка действия мыши 

    elif event == 'graph_mouse_down':
      # Очищаю холст, чтобы не мешали остальные размеченные объекты
      graph.erase()
      graph.draw_image(data=data, location=(-dx,-dy))  

    elif event == 'graph_mouse_drag':
      if Marking:
        x,y = values['graph']
        graph.delete_figure(rectangle)
        rectangle = graph.draw_rectangle((Rect_x0, Rect_y0), (x, y),
                                         fill_color = None,
                                         line_color = ObjectsData.iloc[0,3+Current],
                                         line_width = line_width)
      else:
        # Очищаю холст, чтобы не мешали остальные размеченные объекты
        graph.erase()
        graph.draw_image(data=data, location=(-dx,-dy))        
        Rect_x0, Rect_y0 = values['graph']
        rectangle = graph.draw_rectangle((Rect_x0, Rect_y0), (Rect_x0, Rect_y0),
                                          fill_color = None,
                                          line_color = ObjectsData.iloc[0,3+Current],
                                          line_width = line_width)
        Marking = True
  
    elif event == 'graph_mouse_up':
      if Marking:
        Marking = False
        if Rect_x0 < x:
          ObjectsData.iloc[FileNumber,Current+3][0] = Rect_x0+dx+line_width
          ObjectsData.iloc[FileNumber,Current+3][2] = x+dx-line_width
        else:
          ObjectsData.iloc[FileNumber,Current+3][0] = x+dx-line_width
          ObjectsData.iloc[FileNumber,Current+3][2] = Rect_x0+dx+line_width
        if Rect_y0 < y:
          ObjectsData.iloc[FileNumber,Current+3][1] = Rect_y0-dy+line_width
          ObjectsData.iloc[FileNumber,Current+3][3] = y-dy-line_width
        else:
          ObjectsData.iloc[FileNumber,Current+3][1] = y-dy-line_width
          ObjectsData.iloc[FileNumber,Current+3][3] = Rect_y0-dy+line_width

        # Проверка что разметка не выходит за границу изображение
        if ObjectsData.iloc[FileNumber,Current+3][0] < 0:
          ObjectsData.iloc[FileNumber,Current+3][0] = 0
        if ObjectsData.iloc[FileNumber,Current+3][1] < 0:
          ObjectsData.iloc[FileNumber,Current+3][1] = 0
        if ObjectsData.iloc[FileNumber,Current+3][2] > img_width:
          ObjectsData.iloc[FileNumber,Current+3][2] = img_width
        if ObjectsData.iloc[FileNumber,Current+3][3] > img_height:
          ObjectsData.iloc[FileNumber,Current+3][3] = img_height 

        graph.delete_figure(rectangle)

                  
        # Если в режиме автоматического переключения объектов
        if AutoObjects:
          if Current == MaxObjects-1:
            Current = 0
            Update_Obj_Info()
            if AutoFrames:
              if FileNumber == ObjectsData.shape[0]-1:
                FileNumber =1
              else:
                FileNumber += 1
              FileName = ObjectsData['File'].iloc[FileNumber]
              Redraw_Frame()            
          else:
            Current += 1
            Update_Obj_Info()

      # Поскольку при нажатии объекты очищаются, то когда отпускаем, нужно все восстановить
      Redraw_Objects()        
      x,y = values['graph']
      v_line = graph.draw_line((x,0),(x,height), width = ObjectsData['Width'].iloc[0], color=ObjectsData['Height'].iloc[0])
      h_line = graph.draw_line((0,y),(width,y), width = ObjectsData['Width'].iloc[0], color=ObjectsData['Height'].iloc[0])

    # Drag_n_drop
    elif event == 'graph_midmouse_down':
      graph.set_cursor(cursor='fleur')
    elif event == 'graph_midmouse_up':      
      graph.set_cursor(cursor='left_ptr') 

# Удаление временный файлов
if os.path.exists(DEFAULT_FILE_NAME):
  os.remove(DEFAULT_FILE_NAME)
