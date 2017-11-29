# -*- coding: utf-8 -*-

import cv2 
import copy
import numpy as np 
import dip_lib as dip
import sys
#importar libreria MIDI, cambiar directorio si es necesario
sys.path.append('D:/Documentos/Ces/upc/ciclo 6/Procesamiento de Imagenes/Trabajo Final/Trabajo/MIDIUtil-0.89/src/midiutil')
from MidiFile3 import MIDIFile
MyMIDI = MIDIFile(1)
# carga imagen de partitura
img = dip.load("resource/sheet's/maria.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,gray = cv2.threshold(~img_gray,100,255,cv2.THRESH_BINARY)
# carga de plantillas
b1 = cv2.imread("resource/template's/b1.png",0) 
b2 = cv2.imread("resource/template's/b3.png",0) 
w1 = cv2.imread("resource/template's/w1.png",0) 
w2 = cv2.imread("resource/template's/w2.png",0) 
r1 = cv2.imread("resource/template's/r1.png",0)
r2 = cv2.imread("resource/template's/r2.png",0)

sol = cv2.imread("resource/template's/sol.png",0)
fa = cv2.imread("resource/template's/fa.png",0)

# se hace uso de cv2.matchTemplate para detectar las posiciones (x,y) de las notas deacuerdo a las plantillas
# se etiqueta cada nota colocandole un tiempo segun su tipo
def detectarnota(img,img_gray,nota,color,thres, lista, tiempo):
    w, h = nota.shape[::-1] 
    res = cv2.matchTemplate(img_gray, nota, cv2.TM_CCOEFF_NORMED)
    threshold =thres
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), color, 1)
        lista.append([pt[0],pt[1],tiempo])

def detectarclave(img,img_gray,clave,color,thres,lista):
    w, h = clave.shape[::-1] 
    res = cv2.matchTemplate(img_gray, clave, cv2.TM_CCOEFF_NORMED)
    threshold =thres
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), color, 1)
        lista.append([pt[0],pt[1]])

# al realizar el matchtemplate de cv2 algunas notas son encontras y almacenadas dos veces o incluso con una
# diferencia minima en pixeles tanto en x como en y para ello se utiliza esta funcion
def eliminar_semejantes(arr):
    eliminar = []
    for i in range(np.size(arr,0)):
        for n in range(i + 1,np.size(arr,0)):
            a = abs(arr[i][0] - arr[n][0])
            b = abs(arr[i][1] - arr[n][1])
            if((a < 3) and (b < 3)):
                eliminar.append(n)
    eliminar = np.unique(eliminar)
    res= np.delete(arr,eliminar,0)
    return res

# limpia la imagen y retorna solo las lineas horizontales de la partitura
def Lineas_Horizontales(gray):
    horizontal = np.copy(gray)
    rows,cols = horizontal.shape
    horizontalsize = int (cols/2) # elemento estructural
    Point = (-1,-1)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize,1))
    horizontal = cv2.erode(horizontal, horizontalStructure, Point)
    horizontal= cv2.dilate(horizontal, horizontalStructure, Point)
    horizontal_inv = cv2.bitwise_not(horizontal)
    return horizontal_inv

# revisa verticalmente encontrando y almacenando la altura "y" de las lineas horizontales 
# elimina si es necesario las lineas dobles o gruesas
def listar_notas(horizontal_inv):
    rows,cols = horizontal_inv.shape
    lista = []
    # recorrer dede la mitad de la imagen para encontrar las lineas
    for i in range(0,rows):
        if(horizontal_inv[i][int(cols/2)] == 0):
            lista.append(i)
    m = copy.copy(lista)
    #revisa las dobles lineas
    for j in range(0,len(lista)-1):
        if(lista[j+1] - lista[j] < 5):
            m.pop(j)
    lista = m
    return lista

# divide en partes los pentagramas de la partitura para saber donde inicia
def div_notas(lista):
    n_lista = []
    t_lista = []
    n = int(len(lista)/5)
    for i in range(0,n):
        t_lista = []
        for j in range(5*i,5+5*i):
            t_lista.append(lista[j])
        n_lista.append(t_lista)
    return n_lista

# Agrega las notas existentes entre lineas y fuera del pentagrama
def agre_notas(lista):
    total_lista = []
    for i in range(0, len(lista)):
        n_lista = []
        n_lista.append(min(lista[i]) - int((lista[0][1]-lista[0][0])/2))
        for j in range(0,len(lista[i])):
            n_lista.append(lista[i][j])
            n_lista.append(lista[i][j] + int((lista[0][1]-lista[0][0])/2))
        n_lista.append(max(lista[i]) + int((lista[0][1]-lista[0][0])))
        total_lista.append(n_lista)
    return total_lista

# se agrupan las notas encontradas sin repetidas y con su respectiva etiqueta de tiempo ya sea para 
# negras, blancas o redondas
def Agrupar(notasnegras, notasblancas, notasredondas):
    if(len(notasnegras)>0):
        if(len(notasblancas)>0):
            if(len(notasredondas)>0):
                notas = np.concatenate((notasnegras,notasblancas, notasredondas), axis = 0)
            else:
                notas = np.concatenate((notasnegras,notasblancas), axis = 0)
        elif(len(notasredondas)>0):
            notas = np.concatenate((notasnegras, notasredondas), axis = 0) 
    elif (len(notasblancas)>0):
        if(len(notasredondas)>0):
                notas = np.concatenate((notasblancas, notasredondas), axis = 0)
    notas = sorted(notas, key = lambda order:order[0], reverse = False)
    return notas

# detectar el tipo de clave de la partitura
def definir_clave(clave_sol, clave_fa, clavesol, clavefa):
    if(len(clavesol) > 0):
        return clave_sol
    elif(len(clavefa) > 0):
        return clave_fa

# se analizan las posiciones de las notas tomadas de "Agrupar" para compararla con las posiciones
# de las lineas del pentagrama sacado de "agre_notas"
def Notas(notas, total_lista, clave):
    f_lista = []
    for i in range(0, len(total_lista)):
        for k in range(0,len(notas)):
            for j in range (0, len(total_lista[i])-1):
                if(total_lista[i][j]<=notas[k][1]<total_lista[i][j+1]):
                    f_lista.append([clave[j],notas[k][2]])
    return f_lista

# se genera cada sonido segun la lista de notas
def newNote(time, duration, pitch):
    track = 0
    MyMIDI.addTrackName(track,time,"Sample Track")
    MyMIDI.addTempo(track,time, 120)
    channel = 0
    volume = 100
    MyMIDI.addNote(track,channel,pitch,time,duration,volume)

# se junta los sonido para formar la cancion
def song(f_lista):
    tiempo = 0
    newNote(0,f_lista[0][1],f_lista[0][0])
    for i in range(1,len(f_lista)):
        tiempo = tiempo + f_lista[i-1][1]
        newNote(tiempo,f_lista[i][1],f_lista[i][0])
    binfile = open("FINAL.mid", 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()

# arreglo solo con lineas de la partitura
horizontal_inv = Lineas_Horizontales(gray)
# lista con posiciones en y de las partituras
lista = listar_notas(horizontal_inv)
# se dividen los pentagramas
n_lista = div_notas(lista)
# se agregan notas intermedias
total_lista = agre_notas(n_lista)

#clave sol
clave_sol = [77,76,74,72,71,69,67,65,64,62,60]#[f5,e5,d5,c5,b4,a4,g4,f4,e4,d4,c4]
#clave fa
clave_fa = [69,67,65,64,62,60,59,57,55,53,52]#[a4,g4,f4,e4,d4,c4,b3,a3,g3,f3,e3]
# arreglo de notas detectadas
notasnegras = []
notasblancas = []
notasredondas = []
# arreglo de claves detectadas
clavesol = []
clavefa = []
# detectar y almacenar nota snegras
detectarnota(img,img_gray,b1,(0,0,255),0.65,notasnegras,1)
# detectar y almacenar notas blancas
detectarnota(img,img_gray,w1,(0,255,0),0.70, notasblancas,2)
detectarnota(img,img_gray,w2,(0,255,0),0.70, notasblancas,2)
# detectar y almacenar notas redondas
detectarnota(img,img_gray,r1,(255,0,0),0.70, notasredondas, 4)
detectarnota(img,img_gray,r2,(255,0,0),0.70, notasredondas, 4)
# detectar y almacenar claves
detectarclave(img, img_gray, sol, (255,255,0),0.70,clavesol)
detectarclave(img,img_gray,fa,(255,0,255),0.70,clavefa)
# se elimina notas encontradas varias veces
notasnegras = eliminar_semejantes(notasnegras)
notasblancas = eliminar_semejantes(notasblancas)
notasredondas = eliminar_semejantes(notasredondas)
# se agrupan las notas
notas = Agrupar(notasnegras, notasblancas, notasredondas)
# se define la clave
clave = definir_clave(clave_sol, clave_fa, clavesol, clavefa)
# arreglo con las notas y su tiempo ( etiquetado )
f_lista = Notas(notas,total_lista,clave)    
# imagen con las notas seleccionadas
cv2.imwrite("Notas_Detectadas.jpg",img)
# se crea el archivo midi ( cancion )
song(f_lista)
