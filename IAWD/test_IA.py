from PIL import Image
from IA_Features import IA_Features

img_name = r'../imgs/Beijing_20190530051212642.jpg'
im = Image.open(img_name).convert('RGB')

(width, height) = (im.width // 2, im.height // 2)
im = im.resize((width, height))  # PIL 先 Resize 减少计算量

info = IA_Features().getInfo(im)
print("Extraction of IA Features: ")
print(info)