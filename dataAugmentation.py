from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img, ImageDataGenerator
import matplotlib.pyplot as plt
from numpy import expand_dims
from os import listdir
import timeit
import time
import cv2

# Path da pasta de entrada
localMain = 'E:\CODANDO\FURB\BOBAGENS\Data-Augmentation\entry'
path = (f'{localMain}')
dirs = listdir(path)

start = timeit.default_timer()
for i in range(0, len(dirs)):
    #Carrega imagem para cv2
    imgCV = cv2.imread(f'entry\{dirs[i]}')
    # Carrega para Keras
    imgKR = load_img(f'entry\{dirs[i]}')

    # Flip
    img_flipped = cv2.flip(imgCV, 1)
    cv2.imwrite(f'out\{str(dirs[i])[:-4]}_reverse.jpg', img_flipped)

    # Brilho
    data = img_to_array(imgKR)
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
    it = datagen.flow(samples, batch_size=1)

    for j in range(9):
        batch = it.next()
        imgB = batch[0].astype('uint8')
        imgEnd = plt.imshow(imgB)
        plt.axis('off')
        plt.savefig(f'out\{str(dirs[i])[:-4]}_brilho_{j}.jpg',
                    bbox_inches='tight',
                    pad_inches=0,
                    format='jpg',
                    dpi=140)

    #Zoom
    data = img_to_array(imgKR)
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
    it = datagen.flow(samples, batch_size=1)

    for k in range(9):
        batch = it.next()
        imgB = batch[0].astype('uint8')
        imgEnd = plt.imshow(imgB)
        plt.axis('off')
        plt.savefig(f'out\{str(dirs[i])[:-4]}_zoom_{k}.jpg',
                    bbox_inches='tight',
                    pad_inches=0,
                    format='jpg',
                    dpi=140)

    # Deslocamento
    data = img_to_array(imgKR)
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(width_shift_range=[-200, 200])
    it = datagen.flow(samples, batch_size=1)

    for l in range(9):
        batch = it.next()
        imgB = batch[0].astype('uint8')
        imgEnd = plt.imshow(imgB)
        plt.axis('off')
        plt.savefig(f'out\{str(dirs[i])[:-4]}_dislocada_{l}.jpg',
                    bbox_inches='tight',
                    pad_inches=0,
                    format='jpg',
                    dpi=140)

end = timeit.default_timer()
print('tempo de processamento: %f segundos' % (end - start))