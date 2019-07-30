from preprocess import prepro_api
from incline import incline_api
from segmentation import segment_api
from model import train
from exif import exif_api
import os

RAW_DATA_TRUE = './data/raw_true'
RAW_DATA_FALSE = './data/raw_false'
CHAR_TRUE = './data/char_true'
CHAR_FALSE = './data/char_false'

if __name__ == '__main__':
    # flag = exif_api('./adobe.jpg')
    # flag = exif_api('./data/2.png')

    # prepro_api('./data/2.png', './data/demo_pre.png')
    # rotate_img = incline_api(filename='./data/demo_pre.png', savepath='./data/demo_rota.png')
    # segment_api('./data/demo_rota.png', './data/train', number=20, min_val_word=14)
    # 用原始照片生成单个字照片
    count = 0
    for filename in os.listdir(RAW_DATA_TRUE):
        print(filename)
        file = os.path.join(RAW_DATA_TRUE, filename)
        prepro_api(file, './data/demo_pre.png')
        rotate_img = incline_api(filename='./data/demo_pre.png', savepath='./data/demo_rota.png')
        segment_api('./data/demo_rota.png', CHAR_TRUE, number=count, min_val_word=14)
        count += 3

    count = 0
    for filename in os.listdir(RAW_DATA_FALSE):
        file = os.path.join(RAW_DATA_FALSE, filename)
        prepro_api(file, './data/demo_pre.png')
        rotate_img = incline_api(filename='./data/demo_pre.png', savepath='./data/demo_rota.png')
        segment_api('./data/demo_rota.png', CHAR_FALSE, number=count, min_val_word=14)
        count += 3