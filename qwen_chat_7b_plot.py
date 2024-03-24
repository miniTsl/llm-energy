import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 模型结构文本
model_structure = """
QWenLMHeadModel(
  (transformer): QWenModel(
    (wte): Embedding(151936, 4096)
    (drop): Dropout(p=0.0, inplace=False)
    (rotary_emb): RotaryEmbedding()
    (h): ModuleList(
      (0-31): 32 x QWenBlock(
        (ln_1): RMSNorm()
        (attn): QWenAttention(
          (c_attn): Linear(in_features=4096, out_features=12288, bias=True)
          (c_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (attn_dropout): Dropout(p=0.0, inplace=False)
        )
        (ln_2): RMSNorm()
        (mlp): QWenMLP(
          (w1): Linear(in_features=4096, out_features=11008, bias=False)
          (w2): Linear(in_features=4096, out_features=11008, bias=False)
          (c_proj): Linear(in_features=11008, out_features=4096, bias=False)
        )
      )
    )
    (ln_f): RMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
)
"""

# 饼图数据
qwen_chat_7b = 1588.5527450001239
lm_head = 1686.1715700000525 - qwen_chat_7b
transformer_QWenBlock = 2933.568149999976 - qwen_chat_7b
transformer_QWenAttention = 2013.2381900000573 - qwen_chat_7b
transformer_QWenMLP = 2404.7453999999166 - qwen_chat_7b
transformer_RMSNorm = 1662.6752500000596 - qwen_chat_7b
lm_head_part = lm_head / qwen_chat_7b * 100
QWenBlock_ratio_part = transformer_QWenBlock / qwen_chat_7b * 100
others_ratio_part = 100 - lm_head_part - QWenBlock_ratio_part
QWenAttention_inside_QWenBlock_part = transformer_QWenAttention / transformer_QWenBlock * 100
QWenMLP_inside_QWenBlock_part = transformer_QWenMLP / transformer_QWenBlock * 100
RMSNorm_inside_QWenBlock_part = transformer_RMSNorm / transformer_QWenBlock * 100
others_inside_QWenBlock_part = 100 - QWenAttention_inside_QWenBlock_part - QWenMLP_inside_QWenBlock_part - RMSNorm_inside_QWenBlock_part
print("qwen_chat_7b", qwen_chat_7b)
print("lm_head", lm_head)
print("transformer_QWenBlock", transformer_QWenBlock)
print("transformer_QWenAttention", transformer_QWenAttention)
print("transformer_QWenMLP", transformer_QWenMLP)
print("transformer_RMSNorm", transformer_RMSNorm)

labels = ['lm_head', 'QWenBlock', 'others']
sizes = [lm_head_part, QWenBlock_ratio_part, others_ratio_part]
colors = ['gold', 'yellowgreen', 'lightcoral']

lables_inside_QWenBlock = ['QWenAttention', 'QWenMLP', 'RMSNorm', 'others']
sizes_inside_QWenBlock = [QWenAttention_inside_QWenBlock_part, QWenMLP_inside_QWenBlock_part, RMSNorm_inside_QWenBlock_part, others_inside_QWenBlock_part]
colors_inside_QWenBlock = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

# 创建子图，确保左右子图大小相同
fig, axs = plt.subplots(1, 2, figsize=(24, 8))

# # 左边子图展示模型结构（文字左对齐）
# axs[0].text(-0.2, 0.5, model_structure, horizontalalignment='left', verticalalignment='center', fontsize=16, family='monospace')
# axs[0].axis('off')


# 右边子图展示饼图
axs[0].pie(sizes, labels=labels, colors=colors, radius=1, autopct='%1.1f%%', textprops={'fontsize': 24})
axs[0].axis('equal')
axs[0].set_title('Energy Distribution of QWen-chat-7b', fontsize=24)

axs[1].pie(sizes_inside_QWenBlock, labels=lables_inside_QWenBlock, radius=1, colors=colors_inside_QWenBlock, autopct='%1.1f%%', textprops={'fontsize': 24})
axs[1].axis('equal')
axs[1].set_title('Energy Distribution inside QWenBlock', fontsize=24)

plt.savefig('model_energy_distribution.png')
