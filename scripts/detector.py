import numpy as np
import cv2

def click_values(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixels = frame_hsv[y, x, :]
        print(pixels)


def detector(color_to_detect):
    global frame_hsv
    color_to_detect = color_to_detect.upper()
    # Rangos de Color
    
    lower_range_yellow = np.array([10, 150, 20]) # Rango Inferior Amarillo
    upper_range_yellow = np.array([25, 255, 255]) # Rango Superior Amarillo

    lower_range_blue = np.array([100, 150, 20]) # Rango Inferior Azul
    upper_range_blue = np.array([115, 255, 255]) # Rango Inferior Azul

    lower_range_red = np.array([165, 150, 20]) # Rango Inferior Rojo
    upper_range_red = np.array([180, 255, 255]) # Rango Inferior Rojo

    cv2.namedWindow('Camera')
    cv2.setMouseCallback('Camera', click_values)
    cap = cv2.VideoCapture(2)

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
            if len(cnts_red) != 0:
                for cont in cnts_red:
                    x, y, w ,h  = cv2.boundingRect(cont)
                    if h > 50 and w > 50:
                        # TODO: Mover el brazo
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
                    x, y, w ,h  = cv2.boundingRect(cont)
                    if h > 50 and w > 50:
                            # TODO: Mover el brazo
                        print(f'x={x}, y={y}')
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('Salio del While')


if __name__ == '__main__':
    color = input('Ingresar Color:')
    detector(color)