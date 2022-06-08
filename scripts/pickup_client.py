#!/usr/bin/env python3

from lebron_bot_2.srv import *
from lebron_bot_2.msg import *
import numpy as np
import cv2
import rospy
import sys
import time

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
            xt='w,x,w,x,w,x,w,x,w,x,w,x,w,x,a,x,a,x,w,x,w,x,w,x,w,x,w,x,a,x,w,x,w,x,w,x,w,x,w,x,d,x,w,x,w,x,w,x,w'
            A='w,x,w,x,w,x,w,x,w,x'
            #txt='a,x,a'
            c1=txt.split(',')
            c2=A.split(',')
   
            for c in c1:
        
                callback_move(c,80,45,0.50,0.5)
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
    message = "Hola" # Guardar texto reconocido en esta variable

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

    while True:
        _, frame = cap.read()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(frame_hsv, lower_range_red, upper_range_red)
        mask_yellow = cv2.inRange(frame_hsv, lower_range_yellow, upper_range_yellow)
        mask_blue = cv2.inRange(frame_hsv, lower_range_blue, upper_range_blue)

        if color_to_detect == "R" or color_to_detect == "RED" or color_to_detect == "ROJO":
            # Detect Contours
            cnts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(cnts_red) != 0:
                for cont in cnts_red:
                    if cv2.contourArea(cont) > 500:
                        x, y, w ,h  = cv2.boundingRect(cont)
                        if h > 50 and w > 50:
                            # TODO: Mover el brazo
                            print(f'x={x}, y={y}')
                            if x <= 140 and x > 100 and y <= 140 and y > 100:
                                print("Para")
                            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
        
        elif color_to_detect == "Y" or color_to_detect == "YELLOW" or "AMARILLO":
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
        time.sleep(0.5)

    elif comando=='w':
        Giro_Contra_Motor_B()
        Giro_Contra_Motor_A()
        pwm_a.ChangeDutyCycle(o)
        pwm_b.ChangeDutyCycle(p)
        time.sleep(0.2)

    
    elif comando=='d':
        #PWM_Ang = -1*PWM_Ang
        Giro_Favor_Motor_B()
        Giro_Contra_Motor_A()
        pwm_a.ChangeDutyCycle(PWM_Ang)
        pwm_b.ChangeDutyCycle(PWM_Ang)
        time.sleep(h)



    elif comando=='a':
        Giro_Contra_Motor_B()
        Giro_Favor_Motor_A()
        pwm_a.ChangeDutyCycle(PWM_Ang)
        pwm_b.ChangeDutyCycle(PWM_Ang)
        time.sleep(k)

    else:
        pwm_a.ChangeDutyCycle(0)
        pwm_b.ChangeDutyCycle(0)
        time.sleep(0.5)
        



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

    # Solicito el servicio de navegación
    print(pickup_service_client(group_number, robot_name))

    # Espero 5 Segundos tiempo
    rospy.sleep(5)

    # Solicito el servicio de recoger bolita
    print(ball_service_client('ready_to_pick'))

    # Espero 5 Segundos tiempo
    #rospy.sleep(5)

    # Solicito el servicio de autenticación
    print(launch_auth_client())
    


