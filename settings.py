import torch
# import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNK = 0
PAD = 1

BATCH_SIZE = 256

TRAIN_FILE = 'MachineTranslation/data/train.txt'  # 训练集
VALID_FILE = 'MachineTranslation/data/valid.txt'  # 验证集
TEST_FILE = 'MachineTranslation/data/test.txt'  # 测试文件
SAVE_FILE = 'MachineTranslation/save/model.pth'  # 模型保存路径(注意如当前目录无save文件夹需要自己创建)

LAYERS = 6  # encoder和decoder层数
D_MODEL = 512  # embedding 维度
D_FF = 2048  # feed forward第一个全连接层维数
H_NUM = 8  # multi head attention hidden个数
DROPOUT = 0.1  # dropout比例
EPOCHS = 200
MAX_LENGTH = 60
SRC_VOCAB = 5495  # 英文的单词数
TGT_VOCAB = 2519  # 中文的单词数
step = 0 #tensorboard绘图

# 这里针对的是Valid文件
BLEU_REFERENCES = "MachineTranslation/data/bleu/references.txt" # BLEU评价参考译文
BLEU_REFERENCES_en = "MachineTranslation/data/bleu/references_en.txt" # BLEU评价参考译文对应英文
BLEU_CANDIDATE = "MachineTranslation/data/bleu/candidate.txt"  # 模型翻译译文