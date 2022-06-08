#!/usr/bin/env python3
import time
#from geometry_msgs.msg import Twist
import RPi.GPIO as GPIO

time.sleep(1.5)

#Define nombre de las entradas del puente H
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

#funciones de los motores
def Adelante():
    GPIO.output(in2,True)
    GPIO.output(in1,False)
    GPIO.output(in3,True)
    GPIO.output(in4,False)

def Reversa():
    GPIO.output(in2,False)
    GPIO.output(in1,True)
    GPIO.output(in3,False)
    GPIO.output(in4,True)

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
        
    


def listener():


    #txt='w,x,w,x,w,x,a,x,a,x,w,x,w'
    txt='w,x,w,x,w,x,w,x,w,x,w,x,w,x,a,x,a,x,w,x,w,x,w,x,w,x,w,x,a,x,w,x,w,x,w,x,w,x,w,x,d,x,w,x,w,x,w,x,w'
    A='w,x,w,x,w,x,w,x,w,x'
    #txt='a,x,a'
    c1=txt.split(',')
    c2=A.split(',')
   
    for c in c1:
        
        callback_move(c,80,45,0.50,0.5)
    for c in c2:
        
        callback_move(c,100,40,0.50,0.4)


if __name__ == '__main__':
    listener()




