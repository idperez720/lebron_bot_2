#!/usr/bin/env python3

from tokenize import String
from lebron_bot_2.srv import *
from lebron_bot_2.msg import *
import numpy as np
import cv2
import rospy
import sys

import argparse
import json
import numpy as np
import os
from typing import Tuple, List
import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor

from gpiozero import Servo
import RPi.GPIO as GPIO


# Servicio de Navegación
def pickup_service_client(group_number, robot_name):
    rospy.wait_for_service('pickup_service')
    try:
        pickup_zone_service = rospy.ServiceProxy('pickup_service', pickup_service)
        
        resp = pickup_zone_service(int(group_number), robot_name)
        # Esta variable es suceptible a cambio (Depende del codigo dado por el profesor)
        state = resp.answer.state
        print(state)
        if state == 1:

            # Zona de recogida objetivo
            pickup_zone = resp.answer.pickup_zone # Esta variable es suceptible a cambio (Depende del codigo dado por el profesor)   

            # Coordenadas zona de recogida
            # Esta variable es suceptible a cambio (Depende del codigo dado por el profesor)
            pickup_zone_coordinate = np.array([resp.answer.pickup_zone_x, resp.answer.pickup_zone_y])
            print(pickup_zone_coordinate)
            # TODO: ejecutar codigo de navegación

            # IMPORTANTE!!! NO SALIR DE AQUI HASTA ESTAR SEGURO DE ESTAR EN LA ZONA DE RECOGIDA
            txt='w,x,w,x,w,x,w,x,w,x,w,x,w,x,a,x,a,x,w,x,w,x,w,x,w,x,w,x,a,x,w,x,w,x,w,x,w,x,w,x,d,x,w,x,w,x,w,x,w'
            A='w,x,w,x,w,x,w,x,w,x'
            #txt='a,x,a'
            c1=txt.split(',')
            c2=A.split(',')
   
            for c in c1:
        
                callback_move(c,80,45,0.50,0.5)
            pwm_a.ChangeDutyCycle(0)
            pwm_b.ChangeDutyCycle(0)
            #for c in c2:
        
             #   callback_move(c,100,40,0.50,0.4)
            # IMPORTANTE!!! SE DEBE PUBLICAR LA POSICION EN TOPICO 'robot_position'
            
        return resp
    except rospy.ServiceException as e:
        print(e)

# Servicio de Recogida
def ball_service_client(ready_to_pick):
    rospy.wait_for_service('ball_service')
    try:
        ball_launch_service = rospy.ServiceProxy('ball_service', ball_service)
        resp = ball_launch_service(ready_to_pick)
        
        # Esta variable es suceptible a cambio (Depende del codigo dado por el profesor)
        color = resp.answer.ball_color
        print(resp)
        # Inicia la detección
        detector(color)
    except rospy.ServiceException as e:
        print(e)

# Servicio de autenticación 
def launch_auth_client():

    # TODO: Ejecutar codigo de reconocimiento mensaje
    Mask()
    messageA = main()
    message = messageA # Guardar texto reconocido en esta variable

    rospy.wait_for_service('launch_auth_service')
    try:
        launch_ball_auth_service = rospy.ServiceProxy('launch_auth_service', launch_auth_service)
        resp = launch_ball_auth_service(message)

        # Esta variable es suceptible a cambio (Depende del codigo dado por el profesor)
        auth = resp.auth
        if auth == 1:
            pass
            # TODO: ejecutar codigo de lanzamiento
            
        return resp

    except rospy.ServiceException as e:
        print(e)




################################################################
## BLOQUE DE FUNCIONES AUXILIARES
################################################################
def detector(color_to_detect):

    color_to_detect = color_to_detect.upper()
    # Rangos de Color
    
    lower_range_yellow = np.array([20, 150, 20]) # Rango Inferior Amarillo
    upper_range_yellow = np.array([35, 255, 255]) # Rango Superior Amarillo

    lower_range_blue = np.array([80, 150, 20]) # Rango Inferior Azul
    upper_range_blue = np.array([130, 255, 255]) # Rango Inferior Azul

    lower_range_red = np.array([160, 150, 20]) # Rango Inferior Rojo
    upper_range_red = np.array([180, 255, 255]) # Rango Inferior Rojo

    cap = cv2.VideoCapture(0)

    # Revisar cuando cerrar este bucle
    Escanear=True
    while Escanear:
        _, frame = cap.read()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(frame_hsv, lower_range_red, upper_range_red)
        mask_yellow = cv2.inRange(frame_hsv, lower_range_yellow, upper_range_yellow)
        mask_blue = cv2.inRange(frame_hsv, lower_range_blue, upper_range_blue)

        if color_to_detect == "R" or color_to_detect == "RED" or color_to_detect == "ROJO":
            # Detect Contours
            cnts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rospy.sleep(1)
            if len(cnts_red) != 0:
                for cont in cnts_red:
                    if cv2.contourArea(cont) > 500:
                        x, y, w ,h  = cv2.boundingRect(cont)
                        if h > 50 and w > 50:
                            # TODO: Mover el brazo
                            moveMotor('d',1,0,0,0,5)
                            print(f'x={x}, y={y}')
                            if x <= 140 and x > 100 and y <= 140 and y > 100:
                                
                                print("Para")
                                Escanear=False
                            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
        
        elif color_to_detect == "Y" or color_to_detect == "YELLOW" or color_to_detect == "AMARILLO":
            # Detect Contours
            cnts_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(cnts_yellow) != 0:
                for cont in cnts_yellow:
                    if cv2.contourArea(cont) > 500:
                        x, y, w ,h  = cv2.boundingRect(cont)
                        if h > 50 and w > 50:
                             # TODO: Mover el brazo
                            print(f'x={x}, y={y}')
                            if x <= 140 and x > 100 and y <= 140 and y > 100:
                                print("Para")
                            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 3)
        else:
            # Detect Contours
            cnts_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(cnts_blue) != 0:
                for cont in cnts_blue:
                    if cv2.contourArea(cont) > 500:
                        x, y, w ,h  = cv2.boundingRect(cont)
                        if h > 50 and w > 50:
                             # TODO: Mover el brazo
                            print(f'x={x}, y={y}')
                            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
    
    print('Salio del While')
################################################################
#FUnciones Planeacion
################################################################

ena = 18			
in1 = 23
in2 = 24

enb = 19
in3 = 6
in4 = 5

#configura los pines segun el microprocesador Broadcom
GPIO.setmode(GPIO.BCM)
#configura los pines como salidas
GPIO.setup(ena, GPIO.OUT)
GPIO.setup(enb, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
#Define las salidas PWM q
pwm_a = GPIO.PWM(ena,500)
pwm_b = GPIO.PWM(enb,500)
#inicializan los PWM con un duty Cicly de cero
pwm_a.start(0)
pwm_b.start(0)



def Giro_Favor_Motor_A():
    GPIO.output(in1,True)
    GPIO.output(in2,False)


def Giro_Contra_Motor_A():
    GPIO.output(in1,False)
    GPIO.output(in2,True)


def Giro_Favor_Motor_B():
    GPIO.output(in3,False)
    GPIO.output(in4,True)


def Giro_Contra_Motor_B():
    GPIO.output(in3,True)
    GPIO.output(in4,False)


def callback_move(comando,o,p,k,h):
    velLinA = 17
    velLinB = 17
    velAng = 8

    PWM_LinA = velLinA*100/56 if velLinA < 33 else 100
    PWM_LinB = velLinB*100/56 if velLinB < 33 else 100
    PWM_Ang = velAng*100/16 if velAng < 16 else 100

    if comando=='s':
        #PWM_Lin = -1*PWM_Lin
        Giro_Favor_Motor_A()
        Giro_Favor_Motor_B() 
        pwm_a.ChangeDutyCycle(PWM_LinA)
        pwm_b.ChangeDutyCycle(PWM_LinB)
        rospy.sleep(0.5)

    elif comando=='w':
        Giro_Contra_Motor_B()
        Giro_Contra_Motor_A()
        pwm_a.ChangeDutyCycle(o)
        pwm_b.ChangeDutyCycle(p)
        rospy.sleep(0.2)

    
    elif comando=='d':
        #PWM_Ang = -1*PWM_Ang
        Giro_Favor_Motor_B()
        Giro_Contra_Motor_A()
        pwm_a.ChangeDutyCycle(PWM_Ang)
        pwm_b.ChangeDutyCycle(PWM_Ang)
        rospy.sleep(h)



    elif comando=='a':
        Giro_Contra_Motor_B()
        Giro_Favor_Motor_A()
        pwm_a.ChangeDutyCycle(PWM_Ang)
        pwm_b.ChangeDutyCycle(PWM_Ang)
        rospy.sleep(k)

    else:
        pwm_a.ChangeDutyCycle(0)
        pwm_b.ChangeDutyCycle(0)
        rospy.sleep(0.5)


################################################################
#Funciones Manipulador
################################################################

global ActualA
global ActualB
global ActualC
global ActualD

InicialA=120
InicialB=40
InicialC=130
InicialD=0

# Definicion de los servos

myGPIO1=25
myGPIO2=8
myGPIO3=1
myGPIO4=7

global maxPW
global minPW

myCorrection=0.45
maxPW=(2.0+myCorrection)/1000
minPW=(1.0-myCorrection)/1000

servoA = Servo(myGPIO1,min_pulse_width=minPW,max_pulse_width=maxPW)
#servoA.value=None
servoB = Servo(myGPIO2,min_pulse_width=minPW,max_pulse_width=maxPW)
servoC = Servo(myGPIO3,min_pulse_width=minPW,max_pulse_width=maxPW)
servoD = Servo(myGPIO4,min_pulse_width=minPW,max_pulse_width=maxPW)


ActualA=InicialA
ActualB=InicialB
ActualC=InicialC
ActualD=InicialD

#LLAMAR INICIO

def setPose(A,B,C,D):
    servoA.value=convertirAngulo(A)
    servoB.value=convertirAngulo(B)
    servoC.value=convertirAngulo(C)
    servoD.value=convertirAngulo(D)

def convertirAngulo(angulo):

    return (-1+(angulo*(1/90)))

def arriba():
    moveMotor("b",1,10,10,10)
    moveMotor("c",-1,10,10,10)

def abajo():
    moveMotor("b",-1,10,10,10)
    moveMotor("c",1,10,10,10)

def adelante():
    moveMotor("b",1,10,10,10)
    moveMotor("c",1,10,10,10)

def atras():
    moveMotor("b",-1,10,10,10)
    moveMotor("c",-1,10,10,10)
def abrir():
    servoD.value=convertirAngulo(30)
def cerrar():
    servoD.value=convertirAngulo(0)


def moveMotor(motor,dire,difA,difB,difC,difD):
    global ActualA
    global ActualB
    global ActualC
    global ActualD
    global maxPW
    global minPW
    if motor== 'a':
        angulo=ActualA+dire*(difA)
        
        


        if angulo < 0:
            angulo =0
        if angulo > 180:
            angulo =180
        ActualA=angulo
        print("angulo A:")
        print(ActualA)
        #print(convertirAngulo(angulo))

        servoA.value=convertirAngulo(angulo)

    if motor== 'b':

        angulo=ActualB+dire*(difB)


        if angulo < 20:
            angulo =20
        if angulo > 150:
            angulo =150
        ActualB=angulo
        print("angulo B:")
        print(ActualB)

        servoB.value=convertirAngulo(angulo)
    if motor== 'c':

        angulo=ActualC+dire*(difC)


        if angulo < 110:
            angulo =110
        if angulo > 180:
            angulo =180
        ActualC=angulo
        print("angulo C:")
        print(ActualC)
        servoC.value=convertirAngulo(angulo)


    if motor== 'd':

        angulo=ActualD+dire*(difD)


        if angulo < 0:
            angulo =0
        if angulo > 180:
            angulo =180
        ActualD=angulo
        print("angulo D:")
        print(ActualD)
        servoD.value=convertirAngulo(angulo)
    if motor=='i':
        arriba()
    if motor=='k':
        abajo()
    if motor=='j':
        adelante()
    if motor=='l':
        atras()
    if motor=='o':
        abrir()
    if motor=='p':
        cerrar()

##################################################################
#Funciones Lector
##################################################################
class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best validation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def infer(model: Model, fn_img: Path) -> String:

    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    return(f'"{recognized[0]}"')
    #print(f'Probability: {probability[0]}')


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()

def Mask():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    def Contrastador(imagen):

        # img = cv2.imread('imatext.png', IMREAD_GRAYSCALE)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imageArray=np.array(gray)
        fil= int(imageArray.shape[0])
        filmed = int(imageArray.shape[0]/2)
        col = int(imageArray.shape[1])
    
        columna=0
        flipedrArray=np.fliplr(imageArray)
        flipudArray=np.flipud(imageArray)
        for i, j in np.ndindex(imageArray.shape):
            if imageArray[i][j]>90:
                imageArray[i, j] = 255
            if imageArray[i][j]<90:
                imageArray[i, j] = 0
    
        for j in range(imageArray.shape[1]):
        
            if (imageArray[:,j]-imageArray[:,-j+3]).any()!=0:
                columna =j
                print(columna)
                break
        for j in range(imageArray.shape[1]):
        
            if (flipedrArray[:,j]-flipedrArray[:,j-3]).any()!=0:
                columna1 =j
                print(columna1)
                break

        for i in range(imageArray.shape[0]):
        
            if (imageArray[i,:]-imageArray[i-3,:]).any()!=0:
                fila =i
                print(fila)
                break
        for i in range(imageArray.shape[0]):
        
            if (flipudArray[i,:]-flipudArray[i-3,:]).any()!=0:
                fila1 =i
                print(fila1)
                break
    
        filaNew = int((fila)-20)
        fila1New = int((fila1)-20)
        rangefil=int(fil-fila1New)
        columnaNew=int(columna-20)
        columna1New=int(columna1-20)
        rangecol=int(col-columna1New)
    
        crop_img = imageArray[filaNew:rangefil, columnaNew:rangecol]
        kernel = np.ones((3, 3), np.uint8)
        imgMorph = cv2.erode(crop_img, kernel, iterations = 1)
    


       
    # lower_gray = np.array([0, 0, 0], np.uint8)
    # upper_gray = np.array([179, 50, 230], np.uint8)
    # mask_gray = cv2.inRange(imageArray, lower_gray, upper_gray)
    # img_res = cv2.bitwise_and(img, img, mask = mask_gray)
        cv2.imshow('Logo OpenCV',imgMorph)
        path = '/home/juan/Documents/MyCode/TextdetecTens/SimpleHTR/data'
        cv2.imwrite(os.path.join(path , 'word.png'), imgMorph)
    #cv2.imwrite('im.png', imageArray)
    
        t = cv2.waitKey(1)
    k=0
    while True:
        ret, frame = cap.read()
        if ret==False:break
        doc = frame
        cv2.imshow("Lector inteligente", frame)
        t = cv2.waitKey(1)
        if t==27:break
    

    Contrastador(doc)
    cap.release()
    cv2.destroyAllWindows()



def main():
    """Main function."""

    # parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # train the model
    if args.mode == 'train':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

        # when in line mode, take care to have a whitespace in the char list
        char_list = loader.char_list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters and words
        with open(FilePaths.fn_char_list, 'w') as f:
            f.write(''.join(char_list))

        with open(FilePaths.fn_corpus, 'w') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list, decoder_type)
        train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # evaluate it on the validation set
    elif args.mode == 'validate':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        model = Model(char_list_from_file(), decoder_type, must_restore=True)
        validate(model, loader, args.line_mode)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        infer(model, args.img_file)
        return infer


# if __name__ == '__main__':
#     Mask()
#     main()

##################################################################
##################################################################




def usage():
    return f'{sys.argv[0]}'

if __name__ == '__main__':
    if len(sys.argv) == 3:
        group_number = sys.argv[1]
        robot_name = sys.argv[2]
    else:
        print(usage())
        sys.exit(1)
    print(f"Requesting {group_number, robot_name}")

    setPose(InicialA,InicialB,InicialC,InicialD)

    # Solicito el servicio de navegación
    #print(pickup_service_client(group_number, robot_name))

    # Espero 5 Segundos tiempo
    #rospy.sleep(5)

    # Solicito el servicio de recoger bolita
    #print(ball_service_client('ready_to_pick'))

    # Espero 5 Segundos tiempo
    #rospy.sleep(5)

    # Solicito el servicio de autenticación
    print(launch_auth_client())
    


