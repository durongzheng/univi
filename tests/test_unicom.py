import pytest
import logging
import torch.nn as nn
import torch
import os
from copy import deepcopy

from univi import YOLO
from univi.utils import LOGGER, colorstr
from univi.nn.modules import MultiLabelClassify
from univi.nn import parse_model, yaml_model_load

PT_PTH = '/home/drz/mycode/univi/runs/multi_classify/train43/weights/best.pt'
TASK = 'multi_classify'
IMAGE_SIZE = 448
TEST_LOG_FILE = '/home/drz/mycode/univi/runs/tests/test.log' 
TEST_PT_FILE = '/home/drz/mycode/univi/runs/tests/best.pt'
TEST_DATA_ROOT = '/media/Elements/zipImg/ysh5/test'

def test_load_save_model():
    model = YOLO(TEST_PT_FILE, task=TASK)
    LOGGER.info(f'Model Author: {colorstr("blue", model.ckpt["author"])}')
    LOGGER.info(f'Model License: {colorstr("blue", model.ckpt["license"])}')
    LOGGER.info(f'Model Time: {colorstr("blue", model.ckpt["date"])}')
    LOGGER.info(f'Model Version: {colorstr("blue", model.ckpt["version"])}')
    # model.save(TEST_PT_FILE)

@pytest.mark.skip('暂停')
def test_model_predict():
    model = YOLO(TEST_PT_FILE, task=TASK)
    classes = ['closed', 'cover', 'glasses', 'normal', 'squint', 'vague']

    for i, c in enumerate(classes):
        img_path = os.path.join(TEST_DATA_ROOT, c)
        results = model(source=img_path, imgsz=IMAGE_SIZE)
        false_results = 0
        file_handler = logging.handlers.RotatingFileHandler(TEST_LOG_FILE, mode='a', maxBytes=16777216, backupCount=10, encoding='utf-8')
        LOGGER.addHandler(file_handler)
        for result in results:
            probs = result.probs
            top1 = probs.top1
            if(top1 != i):
                false_results += 1
                LOGGER.info(f'{result.path}, 识别结果: {classes[top1]}, 置信度: {probs.data[top1]}')
        LOGGER.info(f'{c} 总量: {len(results)}, 错误量: {false_results}')
        LOGGER.removeHandler(file_handler)

CFG_FILE = "/home/drz/mycode/univi/univi/cfg/models/v8/unicom-RepViT-multi-cls.yaml"
# CFG_FILE = "/home/drz/mycode/univi/univi/cfg/models/v8/yolov8x.yaml"
def test_parse_model():
    d = yaml_model_load(CFG_FILE)
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]): 
        LOGGER.info(f"Index:{i} \t f:{f} \t n:{n}\t m:{m} \t args:{args} \n")

    model, save = parse_model(d, ch=3, verbose=True)  # model, savelist
    LOGGER.info(f"Save Layers List: {save}")

    for mo in model:
        LOGGER.info(f"index:{str(mo.i):<20} f:{str(mo.f):<25} type:{mo.type}\n")
