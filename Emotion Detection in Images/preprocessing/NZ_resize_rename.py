# Authors: D. Wen, A. Romriell, J. Pollard

from PIL import Image  # Python Imaging Library
import PIL
import glob

"""
Takes all images from the specified directory ('path')
Renames and scales down the image to 48x48 with the defined naming convention
"""

# path for train files
path = '~/EmotionInTheWild/Train/Train_Aligned_Faces/'
save_path = '~/EmotionInTheWild/train_resized_all/'


files = {
    "A": glob.glob(path + "Angry/*.png"),
    "D": glob.glob(path + "Disgust/*.png"),
    "F": glob.glob(path + "Fear/*.png"),
    "H": glob.glob(path + "Happy/*.png"),
    "N": glob.glob(path + "Neutral/*.png"),
    "U": glob.glob(path + "Sad/*.png"),  # unhappy =  sad
    "S": glob.glob(path + "Surprise/*.png")
}
i = 35888  # begin index for train files
# i = 36779  # begin index for val files


for emotion, images in files.iteritems():
    for j in images:
        img = Image.open(j)
        img = img.convert("L")
        img = img.resize((48, 48), PIL.Image.ANTIALIAS)
        save_name = save_path + emotion + '%06d' % i + '.png'
        img.save(save_name)
        i += 1
print i
