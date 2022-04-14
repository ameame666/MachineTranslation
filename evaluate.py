import numpy as np
from torch.autograd import Variable
from infer import greedy_decode, output, sentence2id, src_handle

from utils import bleu_candidate, bleu_references, get_word_dict,subsequent_mask
from settings import *

def evaluate(data, model):
    # 梯度清零
    with torch.no_grad():
        with open(data, 'r', encoding="utf-8") as f_read:
            for line in f_read:
                src,src_mask = src_handle(sentence2id(line))
                out = greedy_decode(model,src,src_mask,max_len=50,start_symbol=int(cn_word2idx.get('BOS')))
                translation = output(out)
                print("English: {}Translation: {}\n".format(line, translation))
                bleu_candidate(" ".join(translation))


def evaluate_test(data, model):
    evaluate(data, model)


if __name__ == '__main__':

    from settings import *
    from train import model
    from data_pre import PrepareData
    model.load_state_dict(torch.load('MachineTranslation/save/model_1.pth', map_location=torch.cuda.set_device(0)))
    cn_idx2word, cn_word2idx, en_idx2word, en_word2idx = get_word_dict()
    # data = PrepareData(TEST_FILE)
    evaluate_test(BLEU_REFERENCES_en, model)