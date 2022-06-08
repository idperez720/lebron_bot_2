#!/usr/bin/env python3
from posixpath import split
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
#import RPi.GPIO as GPIO
from gpiozero import Servo

# Angulos iniciales de cada motor
global difA
global difB
global difC
global difD

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

# Angulo de cada paso


difD=10

# Angulo actual

ActualA=InicialA
ActualB=InicialB
ActualC=InicialC
ActualD=InicialD

# lee la informacion de teleop
print('hola')
def callback_read(data):
    #print("call")


    dato = data.data
    datos=dato.split(',')
    #if datos[0]!="0":
    #    print(datos)
    #Direcion de giro

    dire=int(datos[0])
    difA=int(datos[2])
    difB=int(datos[3])
    difC=int(datos[4])
    
    # Motor

    motor=datos[1]
    moveMotor(motor,dire,difA,difB,difC)

# Convierte el angulo a un valor valido entre -1 y 1

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

def convertirAngulo(angulo):

    return (-1+(angulo*(1/90)))

# Mueve el motor a un angulo

def moveMotor(motor,dire,difA,difB,difC):
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
def inicio():
    servoA.value=convertirAngulo(ActualA)
    servoB.value=convertirAngulo(ActualB)
    servoC.value=convertirAngulo(ActualC)
    servoD.value=convertirAngulo(ActualD)


def listener():
    print("buenas")
    inicio()
    rospy.init_node('robot_listener', anonymous=True)
    rospy.Subscriber('/robot_cmdVel', String, callback_read)
    rospy.spin()


if __name__ == '__main__':
    listener()
