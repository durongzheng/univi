import pytest
import logging
import torch.nn as nn
import torch
import os

from univi import YOLO
from univi.utils import LOGGER, colorstr
from univi.nn.modules import MultiLabelClassify

PT_PTH = '/home/drz/mycode/univi/runs/multi_classify/train42/weights/best.pt'
TASK = 'multi_classify'
IMAGE_SIZE = 448
TEST_LOG_FILE = '/home/drz/mycode/univi/runs/tests/test.log' 
MODEL_STRUCT_FILE = '/home/drz/mycode/univi/runs/tests/best.log'
TEST_PT_FILE = '/home/drz/mycode/univi/runs/tests/best2.pt'
TEST_DATA_ROOT = '/media/Elements/zipImg/ysh5/test'

def test_load_save_model():
    model = YOLO(PT_PTH, task=TASK)
    # file_handler = logging.handlers.RotatingFileHandler(MODEL_STRUCT_FILE, mode='a' , maxBytes=16777216, backupCount=10, encoding='utf-8')
    # LOGGER.addHandler(file_handler)
    # model.model.model[-1] = MultiLabelClassify(c1=256, c2=6)
    # model.model.model[-1].conv.act = nn.SiLU(inplace=True)
    # model.model.model[-1].f = -1
    # model.model.model[-1].i = 14
    # model.model.model[-1].type = 'MultiLabelClassify'
    # model.save(TEST_PT_FILE)
    # LOGGER.info(f'Model Author: {colorstr("blue", model.ckpt["author"])}')
    # LOGGER.info(f'Model License: {colorstr("blue", model.ckpt["license"])}')
    LOGGER.info(f'Model Time: {colorstr("blue", model.ckpt["date"])}')
    LOGGER.info(f'Model Version: {colorstr("blue", model.ckpt["version"])}')

# @pytest.mark.skip('暂停')
def test_model_predict():
    model = YOLO(PT_PTH, task=TASK)
    classes = ['closed', 'cover', 'glasses', 'normal', 'squint', 'vague']

    for i, c in classes:
        img_path = os.path.join(TEST_DATA_ROOT, c)
        results = model(source=img_path, imgsz=IMAGE_SIZE)
        for result in results:
            probs = result.probs
            if(probs.data[1]>=0.15):
                LOGGER.info(f'probs: {probs.data}')
            # LOGGER.info(f'probs sum: {torch.sum(probs.data)}')
