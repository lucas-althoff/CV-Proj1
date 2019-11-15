# Reference: https://github.com/samsammurphy/pixel_selector
# Reference2: https://stackoverflow.com/questions/33853802/mouse-click-events-on-a-live-stream-with-opencv-and-python

import math
import cv2
import numpy as np

print("### Seletor automático de regiões de cores ###")
print("Critérios do sistema => imagem colorida ou em escala de cinza em formato 'jpg'; vídeos apenas coloridos 'em formato 'avi' ou 'h264'")

# mouse callback functions
def videoPixSelect(event,x,y,flags,param):
  if event == cv2.EVENT_LBUTTONDOWN:
    Bc = frame[y,x,0]
    Gc = frame[y,x,1]
    Rc = frame[y,x,2]  
    print("x | y")
    print(x, y)
    print("B | G | R")
    print(Bc, Gc, Rc)
    font = cv2.FONT_HERSHEY_SIMPLEX
    strimg = str(x) + ',' + str(y) + ':' +str(Bc) + ',' + str(Gc) + ',' + str(Rc)
    cv2.putText(frame, strimg, (x,y), font, 0.6, (255,255,0), 2)
    cv2.circle(frame,(x,y),7,(255,255,255),-1)
    cv2.circle(frame,(x,y),5,(0,0,0),-1)
    colorPixClass(frame, Bc, Gc, Rc, fn)

def colorPixSelect(event,x,y,flags,param):
  if event == cv2.EVENT_LBUTTONDOWN:
    Bc = img[y,x,0]
    Gc = img[y,x,1]
    Rc = img[y,x,2]  
    print("x | y")
    print(x, y)
    print("B | G | R")
    print(Bc, Gc, Rc)
    font = cv2.FONT_HERSHEY_SIMPLEX
    strimg = str(x) + ',' + str(y) + ':' +str(Bc) + ',' + str(Gc) + ',' + str(Rc)
    cv2.putText(img, strimg, (x,y), font, 0.6, (255,255,255), 2)
    cv2.circle(img,(x,y),7,(255,255,255),-1)
    cv2.circle(img,(x,y),5,(0,0,0),-1)
    colorPixClass(img, Bc, Gc, Rc, fn)

def colorPixClass(img, Bc, Gc, Rc, fn):
  clone = img.copy()
  h = img.shape[0]
  w = img.shape[1]
  for i in np.arange(h):
    for j in np.arange(w):
      # Definir o pixel com a cor vermelha
      if math.sqrt(math.pow(Bc-img[i,j,0],2) + math.pow(Gc-img[i,j,1],2) + math.pow(Rc-img[i,j,0],2)) <= 13.:
        clone[i,j,0] = 0
        clone[i,j,1] = 0
        clone[i,j,2] = 255
  cv2.namedWindow(fn + "- Regioes selecionadas por cor", cv2.WINDOW_NORMAL)
  cv2.imshow(fn + " - Regioes selecionadas por cor",clone)

def grayPixSelect(event,x,y,flags,param):
  if event == cv2.EVENT_LBUTTONDOWN:
    n = img[y,x]
    print("x | y : Nível de cinza" )
    print(x,y," : ",n)
    font = cv2.FONT_HERSHEY_SIMPLEX
    strimg = str(x) + ', ' + str(y) + ': ' + str(n)
    cv2.putText(img, strimg, (x,y), font, 1, (255,255,0), 2)
    cv2.circle(img,(x,y),7,(255,255,255),-1)
    cv2.circle(img,(x,y),5,(0,0,255),-1) 
    grayPixClass(img,n,fn)

def grayPixClass(img,n,fn):
  clone = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  h = img.shape[0]
  w = img.shape[1]
  for i in np.arange(h):
    for j in np.arange(w):
      if abs(n - img[i,j]) <= 13.:
        clone[i,j,0] = 0
        clone[i,j,1] = 0
        clone[i,j,2] = 255
  cv2.namedWindow(fn + " - Regioes selecionadas por nivel de cinza", cv2.WINDOW_NORMAL)
  cv2.imshow(fn + " - Regioes selecionadas por nivel de cinza",clone)

while True:
  ans = input("Deseja fazer uma analise em imagem (i), video(v) ou em tempo real na webcam (w)?")
  #Confere se é um vídeo, webcam, uma imagem em escala de cinza ou colorida
  if ans == 'w':  
    cam = cv2.VideoCapture(0)
    print("### Estudo em video: Para interagir clique no pixel desejado ###\n")
    #Reading the first frame
    (grabbed, frame) = cam.read()
    while(cam.isOpened()):
      (grabbed, frame) = cam.read()
      fn = "Webcam"
      cv2.namedWindow("Webcam")
      cv2.setMouseCallback('Webcam', videoPixSelect)
      cv2.imshow('Webcam',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break   
    e = input("Deseja analisar novamente (y|n)?")
  elif ans == 'i': 
    fp = input("Digite o caminho do arquivo: ")
    print("SE LIGAAAA", fp)
    img = cv2.imread(fp)
    print("TIPO DA IMG", type(img)) 
    #Confere se o arquivo está dentro dos critérios
    allowed_extenstions = ['jpg']
    if not [fp.lower().endswith(ext) for ext in allowed_extenstions]:
      print('Formato inadequado da imagem')
      break
    fn = input("Digite o nome da imagem: ")
    #Confere se e uma imagem colorida ou em escala de cinza
    img_color = input("É uma imagem colorida (y|n)? ")
    if img_color == 'y':
      img = cv2.imread(fp)
      print("Dimensoes da imagem:")
      print(img.shape)      
      cv2.imshow(fn,img)
      cv2.setMouseCallback(fn, colorPixSelect)
      print("### Estudo de imagem colorida: Para interagir clique no pixel desejado ###\n")
      cv2.waitKey(0)
      e = input("Deseja analisar novamente (y|n)?")
    else:
      img = cv2.imread(fp,0)
      print("Dimensoes da imagem:")
      print(img.shape)
      cv2.imshow(fn,img) 
      cv2.setMouseCallback(fn, grayPixSelect)
      print("### Estudo de imagem em nivel de cinza: Para interagir clique no pixel desejado ###\n")
      cv2.waitKey(0)
      e = input("Deseja analisar novamente (y|n)?")
  elif ans == 'v':
    fp = input("Digite o caminho do arquivo: ")
    #Confere se o arquivo está dentro dos critérios
    allowed_extenstions = ['avi', 'h264']
    if not [fp.lower().endswith(ext) for ext in allowed_extenstions]:
      print('Formato inadequado do video')
      break
    fn = input("Digite o nome da imagem: ")
    cap = cv2.VideoCapture(fp)
    print("### Video analysis: Left-Click to select pixel you want ###\n")
    print("Warning: the video is looping")

    #Reading the first frame
    (grabbed, frame) = cap.read()
    frame_counter = 1
    while(cap.isOpened()):
      frame_counter += 1
      #If the last frame is reached, reset the capture and the frame_counter
      if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0 #Or whatever as long as it is the same as next line
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      (grabbed, frame) = cap.read()
      cv2.namedWindow(fn)
      cv2.setMouseCallback('frame', videoPixSelect)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break   
  if e == "n":
    cap.release()
    cv2.destroyAllWindows()
    break
