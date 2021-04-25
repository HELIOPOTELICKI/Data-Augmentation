from keras.preprocessing.image import load_img, img_to_array
from augmentations import Augmentations as au
from numpy import expand_dims
from os import listdir
import timeit


def clearAll():
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]


def main():
    for i in range(len(dirs)):
        prefix = str(dirs[i])[:-4]
        imgKR = load_img(f'{localMain}\{dirs[i]}')
        data = img_to_array(imgKR)
        sample = expand_dims(data, 0)

        newImage = au(out=out, sample=sample, prefix=prefix)
        newImage.brightness()
        clearAll()

        newImage = au(out=out, sample=sample, prefix=prefix)
        newImage.zoom()
        clearAll()

        newImage = au(out=out, sample=sample, prefix=prefix)
        newImage.displacement()
        clearAll()


if __name__ == '__main__':
    # Path da pasta de entrada
    localMain = 'E:\CODANDO\FURB\BOBAGENS\Data-Augmentation\entry'
    out = 'out'

    path = (f'{localMain}')
    dirs = listdir(path)
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    print('Pronto, tempo decorrido: %f segundos' % (end - start))
