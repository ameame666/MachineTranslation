import time

from data_pre import PrepareData
from model import make_model
from settings import *
from utils import SimpleLossCompute, LabelSmoothing, NoamOpt
import torch
torch.cuda.set_device(0)

from torch.utils.tensorboard import SummaryWriter

# 模型的初始化
model = make_model(SRC_VOCAB, TGT_VOCAB, LAYERS, D_MODEL, D_FF, H_NUM, DROPOUT)

def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # 里面包含了 计算loss -> backward -> optimizer.step() -> 梯度清零
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens  # 实际的词数

        if i % 50 == 0:
            elapsed = time.time() - start
            print("Loss: %f / Tokens per Sec: %fs" % (loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(train_data, valid_data, model, criterion, optimizer):
    #使用SummaryWriter绘制损失函数图像
    writer = SummaryWriter("MachineTranslation/log")

    # 初始化模型在valid集上的最优Loss为一个较大值
    best_valid_loss = 1e5
    step = 0
    for epoch in range(EPOCHS):
        print(f"#" * 50 + f"Epoch: {epoch + 1}" + "#" * 50)
        print('>Train')
        model.train()
        train_loss = run_epoch(train_data.data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        writer.add_scalar("train_loss", train_loss, step)
        model.eval()
        # 在valid集上进行loss评估
        print('>Evaluate')
        valid_loss = run_epoch(valid_data.data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        writer.add_scalar("valid_loss", valid_loss, step)
        step += 1
        print('Evaluate loss: %f' % valid_loss)
        # 如果当前epoch的模型在valid集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), SAVE_FILE)
            best_valid_loss = valid_loss
            print('> Save model done...')
        print()


if __name__ == '__main__':
    # data preprocessing'
    train_data = PrepareData(TRAIN_FILE)
    valid_data = PrepareData(VALID_FILE)
    # training part
    criterion = LabelSmoothing(TGT_VOCAB, padding_idx=0, smoothing=0.1)  # 损失函数
    optimizer = NoamOpt(D_MODEL, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))  # 优化器
    train(train_data, valid_data, model, criterion, optimizer)  # 训练函数(含保存)

    # print(model)
