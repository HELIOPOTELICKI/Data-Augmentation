from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
from numpy import expand_dims
from os import listdir
import timeit
import cv2

# Path da pasta de entrada
localMain = 'E:\CODANDO\FURB\BOBAGENS\Data-Augmentation\entry'
path = (f'{localMain}')
dirs = listdir(path)

start = timeit.default_timer()
for i in range(0, len(dirs)):
    # prefixo para salvamento
    prefix = str(dirs[i])[:-4]

    # Carrega imagem para cv2
    imgCV = cv2.imread(f'entry\{dirs[i]}')

    # Carrega imagem para Keras
    imgKR = load_img(f'entry\{dirs[i]}')
    data = img_to_array(imgKR)
    samples = expand_dims(data, 0)

    # Inverte
    img_flipped = cv2.flip(imgCV, 1)
    cv2.imwrite(f'out\{prefix}_reverse.jpg', img_flipped)

    # Brilho
    datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
    it = datagen.flow(samples, batch_size=1)

    for j in range(9):
        batch = it.next()
        imgB = batch[0].astype('uint8')
        imgEnd = plt.imshow(imgB)
        plt.axis('off')
        plt.savefig(f'out\{prefix}_brilho_{j}.jpg',
                    bbox_inches='tight',
                    pad_inches=0,
                    format='jpg',
                    dpi=140)

    # Zoom
    datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
    it = datagen.flow(samples, batch_size=1)

    for k in range(9):
        batch = it.next()
        imgB = batch[0].astype('uint8')
        imgEnd = plt.imshow(imgB)
        plt.axis('off')
        plt.savefig(f'out\{prefix}_zoom_{k}.jpg',
                    bbox_inches='tight',
                    pad_inches=0,
                    format='jpg',
                    dpi=140)

    # Deslocamento
    datagen = ImageDataGenerator(width_shift_range=[-200, 200])
    it = datagen.flow(samples, batch_size=1)

    for l in range(9):
        batch = it.next()
        imgB = batch[0].astype('uint8')
        imgEnd = plt.imshow(imgB)
        plt.axis('off')
        plt.savefig(f'out\{prefix}_dislocada_{l}.jpg',
                    bbox_inches='tight',
                    pad_inches=0,
                    format='jpg',
                    dpi=140)

end = timeit.default_timer()
print('Tempo de processamento: %f segundos' % (end - start))