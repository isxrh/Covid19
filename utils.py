import torch
import logging

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def set_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    hterm = logging.StreamHandler()  # 输出到终端
    hfile = logging.FileHandler(filename)  # 输出到文件
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    hterm.setFormatter(formatter)
    hfile.setFormatter(formatter)
    logger.addHandler(hterm)
    logger.addHandler(hfile)
    return logger
