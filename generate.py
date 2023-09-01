import os, sys, torch
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
from rwkv_numba.model import RWKV # pip install rwkv
model = RWKV(model='D:/ChatRWKV-main/model/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096', strategy='cuda fp16')

from rwkv_numba.utils import PIPELINE, PIPELINE_ARGS
#pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


args = PIPELINE_ARGS(temperature=1.2, top_p=0.2, top_k=0, # top_k = 0 then ignore
                     alpha_frequency=0.0,
                     alpha_presence=0.0,
                     token_ban=[0], # ban the generation of some tokens
                     token_stop=[], # stop generation whenever you see any token here
                     chunk_len=256) # split input into chunks to save VRAM (shorter -> slower)

########################################################################################################

import time

#msg = "你好"


def gen(msg):
    answer, state = pipeline.generate(msg, token_count=500, args=args)
    print(answer, len(state))
    return answer

msg1 = ["Q: 你是谁？\n\nA:",
        "Q: 你好\n\nA:",
        "Q: 西瓜是什么\n\nA:",
        "Q: 你有多厉害？\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: hi\n\nA:",
        "Q: 企鹅会飞吗？\n\nA:",
        "Q: 企鹅会飞吗？\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 将patient翻译为中文\n\nA:",
        "Q: What\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 运动后不出汗，皮肤发热　运动后不出汗，皮肤发热经常这样没有\n\nA:",
        "Q: What\n\nA:",
        "Q: 你是谁？\n\nA:",
        "Q: 你好\n\nA:",
        "Q: 你好\n\nA:",
        "Q: hi\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: hi\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 你能？\n\nA:",
        ]
msg2 = ["Q: 将patient翻译为中文\n\nA:", "Q: 你好\n\nA:"]
#gen(msg)

gen(msg2)
gen(msg1)


