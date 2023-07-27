import cv2
import PySimpleGUI as sg
from screeninfo import get_monitors
import numpy as np

type_list = (('Видео','*.avi *.mp4 *.mpeg'),)

sg.theme('Dark Red')

Full_Screen = True
Started_Frame = 5000

# Основа для модели нейронной сети
from tensorflow.keras.models import Model 

# Стандартные слои keras
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization 

# Оптимизатор Adam
from tensorflow.keras.optimizers import Adam 

# Параметры обучения нейросети
IMG_W = 640              # Ширина картинки 
IMG_H = 360              # Высота картинки 
CLASS_COUNT = 2               # Количество классов на изображении
path_to_weights = 'Models_Weights/unet_carpets_weights_grayscale.h5'

# Цвета пикселов сегментированных изображений
CARPET = (0, 0, 255)            # Ковер (красный)
OTHER = (0, 0, 0)               # Остальное (черный)
CLASS_LABELS = (OTHER, CARPET)

# Функция преобразования тензора меток класса в цветное сегметрированное изображение
def labels_to_rgb(image_list  # список одноканальных изображений 
                 ):

    result = []

    # Для всех картинок в списке:
    for y in image_list:
        # Создание пустой цветной картики
        temp = np.zeros((IMG_H, IMG_W, 3), dtype='uint8')
        
        # По всем классам:
        for i, cl in enumerate(CLASS_LABELS):
            # Нахождение пикселов класса и заполнение цветом из CLASS_LABELS[i]
            temp[np.where(np.all(y==i, axis=-1))] = CLASS_LABELS[i]

        result.append(temp)
  
    return np.array(result)

# Модель нейронной сети
def unet(class_count,   # количество классов
         input_shape    # форма входного изображения
         ): 
  
    img_input = Input(input_shape)                                          # Создаем входной слой формой input_shape

    ''' Block 1 '''
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)          # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)                                     # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling2D()(block_1_out)                                         # Добавляем слой MaxPooling2D

    ''' Block 2 '''
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)         # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)         # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)                                     # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_2_out)                                         # Добавляем слой MaxPooling2D

    ''' Block 3 '''
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)         # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)         # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)         # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)                                     # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = MaxPooling2D()(block_3_out)                                         # Добавляем слой MaxPooling2D

    ''' Block 4 '''
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)         # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)         # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)         # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    block_4_out = Activation('relu')(x)                                     # Добавляем слой Activation и запоминаем в переменной block_4_out
    x = block_4_out 

    ''' UP 2 '''
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)     # Добавляем слой Conv2DTranspose с 256 нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = concatenate([x, block_3_out])                                       # Объединяем текущий слой со слоем block_3_out
    x = Conv2D(256, (3, 3), padding='same')(x)                              # Добавляем слой Conv2D с 256 нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    ''' UP 3 '''
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)     # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = concatenate([x, block_2_out])                                       # Объединяем текущий слой со слоем block_2_out
    x = Conv2D(128, (3, 3), padding='same')(x)                              # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same')(x)                              # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    ''' UP 4 '''
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)      # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = concatenate([x, block_1_out])                                       # Объединяем текущий слой со слоем block_1_out
    x = Conv2D(64, (3, 3), padding='same')(x)                               # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same')(x)                               # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(class_count, (3, 3), activation='softmax', padding='same')(x)  # Добавляем Conv2D-Слой с softmax-активацией на class_count-нейронов

    model = Model(img_input, x)                                             # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    # Возвращаем сформированную модель
    return model

# Создание модели и вывод сводки по архитектуре
model = unet(CLASS_COUNT,
                  (IMG_H, IMG_W, 1))

#Загружаю веса модели
model.load_weights(path_to_weights)

# Запуск проигрывателя
if Full_Screen:
  for m in get_monitors():
    if m.is_primary:
      MONITOR_WIDTH = m.width
      MONITOR_HEIGHT = m.height

  INTERFACE_X = 10
  INTERFACE_Y = 60

  MAX_WIDTH = MONITOR_WIDTH - INTERFACE_X
  MAX_HEIGHT = MONITOR_HEIGHT - INTERFACE_Y

Path = sg.popup_get_file('Открыть файл', no_window=True, file_types=type_list)
if len(Path) != 0: #Проверка что файл был выбран, а не просто закрыто окно
  cap = cv2.VideoCapture(Path)
  cap.set(cv2.CAP_PROP_POS_FRAMES, Started_Frame)
  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  if Full_Screen:
    Vscale = MAX_WIDTH/width
    Hscale = MAX_HEIGHT/height
    if Vscale < Hscale: Scale_Coef = Vscale/2
    else: Scale_Coef = Hscale/2
    player_width = int(width*Scale_Coef)
    player_height = int(height*Scale_Coef) 
  
  ret, frame = cap.read()
  if Full_Screen:
    frame = cv2.resize(frame, (player_width,player_height), interpolation = cv2.INTER_LINEAR)  
  prevframe = frame    # Первый кадр
  ret, frame = cap.read()
  if Full_Screen:
    frame = cv2.resize(frame, (player_width,player_height), interpolation = cv2.INTER_LINEAR)  
  preprevframe = prevframe
  prevframe = frame
  while True:
      ret, frame = cap.read()
      if ret:
          if Full_Screen:
            frame = cv2.resize(frame, (player_width,player_height), interpolation = cv2.INTER_LINEAR)  
          diff = cv2.absdiff(cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY),cv2.cvtColor(preprevframe, cv2.COLOR_BGR2GRAY))
          coloreddiff = cv2.absdiff(prevframe,preprevframe)
          next_diff  = cv2.absdiff(cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY),cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
          frames3_diff = cv2.absdiff(diff,next_diff)
          # Предсказание
          img_for_predict = cv2.resize(next_diff, (IMG_W,IMG_H), interpolation = cv2.INTER_LINEAR)
          #img_for_predict = cv2.cvtColor(img_for_predict, cv2.COLOR_GRAY2BGR)
          predict = np.argmax(model.predict(np.array([img_for_predict])), axis=-1)
          orig = labels_to_rgb(predict[..., None])
          if Full_Screen:
            orig = cv2.resize(orig[0], (player_width,player_height), interpolation = cv2.INTER_LINEAR) 
          else: orig = cv2.resize(orig[0], (width, height), interpolation = cv2.INTER_LINEAR)
          line_1st = np.hstack((prevframe, orig))
          line_2st = np.hstack((frames3_diff, diff))
          result = np.vstack((line_1st, cv2.cvtColor(line_2st, cv2.COLOR_GRAY2BGR)))
          cv2.imshow('video', result)
          preprevframe = prevframe
          prevframe = frame
          k = cv2.waitKey(30) & 0xff
          if k == 27:
              break
      else:
          break
  cv2.destroyAllWindows()
  cap.release()
