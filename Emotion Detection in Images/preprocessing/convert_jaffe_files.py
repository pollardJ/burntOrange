# Authors: D. Wen, A. Romriell, J. Pollard

from os import listdir
from os.path import isfile, join
import re
from PIL import Image

#convert tiff to png

if __name__ == '__main__':

    id = 37210
    dir = '~/jaffe-japanesefacialexpressions/'

    onlyfiles = [join(dir,f) for f in listdir(dir) if isfile(join(dir, f))]
    tiff_files = [f for f in onlyfiles if f.endswith(".tiff")]

    # KA.AN1.39.png
    # drop first two letters
    # AN -> A   anger
    # DI -> D   disgust
    # FE -> F   fear
    # HA -> H   happy
    # NE -> N   neutral
    # SA -> U   unhappy (previously sad)
    # SU -> S   surprise

    for name in tiff_files:
        im = Image.open(name)
        im = im.resize((48, 48))
        png_name = re.sub("\.tiff$", ".png", name)   # change suffix
        png_name = re.sub("\w\w\.AN\d\.\d\d?\d?", "A%06d"%id, png_name)
        png_name = re.sub("\w\w\.DI\d\.\d\d?\d?", "D%06d"%id, png_name)
        png_name = re.sub("\w\w\.FE\d\.\d\d?\d?", "F%06d"%id, png_name)
        png_name = re.sub("\w\w\.HA\d\.\d\d?\d?", "H%06d"%id, png_name)
        png_name = re.sub("\w\w\.NE\d\.\d\d?\d?", "N%06d"%id, png_name)
        png_name = re.sub("\w\w\.SA\d\.\d\d?\d?", "U%06d"%id, png_name)
        png_name = re.sub("\w\w\.SU\d\.\d\d?\d?", "S%06d"%id, png_name)

        print 'converting %s to %s' % (name, png_name)
        id += 1
        im.save(png_name)