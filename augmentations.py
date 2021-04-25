from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class Augmentations:
    def __init__(self, out, sample, prefix):
        self.out = out
        self.sample = sample
        self.prefix = prefix
        self.dpi = 299

    def brightness(self):
        datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
        it = datagen.flow(self.sample, batch_size=1)

        for j in range(9):
            batch = it.next()
            imgB = batch[0].astype('uint8')
            imgEnd = plt.imshow(imgB)
            plt.axis('off')
            plt.savefig(f'{self.out}\{self.prefix}_brilho_{j}.jpg',
                        bbox_inches='tight',
                        pad_inches=0,
                        format='jpg',
                        dpi=self.dpi)
            plt.cla()

    def zoom(self):
        datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
        it = datagen.flow(self.sample, batch_size=1)

        for k in range(9):
            batch = it.next()
            imgB = batch[0].astype('uint8')
            imgEnd = plt.imshow(imgB)
            plt.axis('off')
            plt.savefig(f'{self.out}\{self.prefix}_zoom_{k}.jpg',
                        bbox_inches='tight',
                        pad_inches=0,
                        format='jpg',
                        dpi=self.dpi)
            plt.cla()

    def displacement(self):
        datagen = ImageDataGenerator(width_shift_range=[-200, 200])
        it = datagen.flow(self.sample, batch_size=1)

        for l in range(9):
            batch = it.next()
            imgB = batch[0].astype('uint8')
            imgEnd = plt.imshow(imgB)
            plt.axis('off')
            plt.savefig(f'{self.out}\{self.prefix}_deslocada_{l}.jpg',
                        bbox_inches='tight',
                        pad_inches=0,
                        format='jpg',
                        dpi=self.dpi)
            plt.cla()