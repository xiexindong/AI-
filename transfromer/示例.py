"""
Transformer 最小可运行实现(零依赖纯 Python 版)
==================================================

目标:不借助 numpy / PyTorch,用最朴素的 Python 代码,一步步展示
Transformer 的核心组件到底在算什么:

  【组件 1】词嵌入 Embedding      —— 把 token 变成数字向量
  【组件 2】位置编码 Positional   —— sin/cos 注入顺序信息
  【组件 3】Self-Attention        —— Q/K/V 算词与词的关系(核心!)
  【组件 4】Multi-Head           —— 多组 Q/K/V 并行,关注多种关系
  【组件 5】FFN + 残差 + LayerNorm —— 逐 token 加工,稳定深层训练

包含 2 个版本:
  【版本 1】ToyTransformer —— 纯 Python 手搓,每一行都能看懂
  【版本 2】PyTorch 真实写法 —— 作为对照(装了 torch 再跑)

注意:这里用「随机初始化的权重」演示计算流程,不是训练好的模型,
所以注意力权重没有真实语义;真实模型是通过海量语料训练,让这些
权重学会「猫追狗」≠「狗追猫」等语义关系的。
"""

import math

# ─────────────────────────────────────────────
# 0. 词表 + 随机数种子(保证结果可复现)
# ─────────────────────────────────────────────

VOCAB = ["我", "爱", "猫", "很", "你"]   # 极简词表
TOKEN2ID = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)               # 5
D_MODEL = 8                           # 嵌入/模型维度
N_HEADS = 2                           # 多头数量
D_K = D_MODEL // N_HEADS              # 每头维度 = 4
SEQ_LEN = 3                           # 演示句子长度


def make_rng(seed: int):
    """极简伪随机数生成器(线性同余),保证任何环境结果一致"""
    state = seed

    def rng():
        nonlocal state
        state = (state * 1103515245 + 12345) % (2**31)
        return state / (2**31)        # 返回 [0,1)

    return rng


# ─────────────────────────────────────────────
# 1. 词嵌入 Embedding(随机初始化的查表)
# ─────────────────────────────────────────────
# 真实:Embedding(vocab_size, d_model) 训练得到。
# 这里:用一个 5x8 的随机表,查 token 对应的那一行。

def make_embed(rng):
    return [[rng() - 0.5 for _ in range(D_MODEL)] for _ in range(VOCAB_SIZE)]


def embed_row(embed, token_id: int):
    return list(embed[token_id])


# ─────────────────────────────────────────────
# 2. 位置编码 Positional Encoding(sin/cos)
# ─────────────────────────────────────────────
# 公式:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

def positional_encoding(seq_len: int, d_model: int):
    pe = []
    for pos in range(seq_len):
        row = []
        for i in range(d_model):
            angle = pos / (10000 ** (2 * i / d_model))
            row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        pe.append(row)
    return pe


# ─────────────────────────────────────────────
# 3. 基础线性代数小工具
# ─────────────────────────────────────────────

def vec_matmul(v, M):
    """向量 v(n) × 矩阵 M(n x p) → 向量(p)"""
    return [sum(v[k] * M[k][j] for k in range(len(v)))
            for j in range(len(M[0]))]


def mat_matmul(A, B):
    """矩阵 A(m x n) × 矩阵 B(n x p) → 矩阵(m x p)"""
    m, p = len(A), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(len(B)))
             for j in range(p)] for i in range(m)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(row):
    """行级 softmax:把分数变成 0~1 的权重,总和 = 1"""
    mx = max(row)
    exps = [math.exp(x - mx) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


# ─────────────────────────────────────────────
# 4. 单头 Self-Attention ⭐(最核心)
# ─────────────────────────────────────────────
# 输入 X: seq_len x d_model (每个 token 一行)
# 流程:
#   Q = X·Wq   (我要找什么)
#   K = X·Wk   (我是什么)
#   V = X·Wv   (我能提供什么信息)
#   scores = Q·Kᵀ / √d_k   (两两打分)
#   weights = softmax(scores)  (归一化成权重)
#   output = weights·V         (按权重融合信息)

def self_attention(X, Wq, Wk, Wv, d_k):
    Q = [vec_matmul(x, Wq) for x in X]
    K = [vec_matmul(x, Wk) for x in X]
    V = [vec_matmul(x, Wv) for x in X]

    # 打分:第 i 个 token 对第 j 个 token 的相关度
    scores = []
    for q in Q:
        scores.append([dot(q, k) / math.sqrt(d_k) for k in K])

    # 归一化:每行 softmax → 权重矩阵(第 i 行 = i 应该从每个 j 吸收多少信息)
    weights = [softmax(row) for row in scores]

    # 加权求和:output[i] = Σ_j weights[i][j] * V[j]
    output = []
    for w in weights:
        output.append([sum(w[t] * V[t][d] for t in range(len(V)))
                       for d in range(len(V[0]))])
    return output, weights


# ─────────────────────────────────────────────
# 5. 多头注意力 Multi-Head
# ─────────────────────────────────────────────
# 标准做法:每头有自己独立的 Wq/Wk/Wv(输出 d_k 维),
# 把 h 个头的输出在最后一维拼接 → d_model 维。
# 效果:不同头可以学「语法关系」「指代关系」「语义相似」等不同模式。

def multi_head_attention(X, heads_weights, d_k):
    head_outputs = []
    for Wq, Wk, Wv in heads_weights:      # 每头独立算
        out, _ = self_attention(X, Wq, Wk, Wv, d_k)
        head_outputs.append(out)

    seq_len = len(X)
    # 拼接:对每个位置 t,把 h 个头在 t 处的输出按顺序接起来
    concat = []
    for t in range(seq_len):
        row = []
        for h_out in head_outputs:
            row.extend(h_out[t])
        concat.append(row)
    return concat


# ─────────────────────────────────────────────
# 6. FFN + LayerNorm + 残差(凑成一个 Encoder Block)
# ─────────────────────────────────────────────

def relu(x):
    return x if x > 0 else 0.0


def layer_norm(row):
    """LayerNorm:对每个 token 自己的所有维度做归一化(和 batch 无关)"""
    mean = sum(row) / len(row)
    var = sum((x - mean) ** 2 for x in row) / len(row)
    return [(x - mean) / math.sqrt(var + 1e-6) for x in row]


def ffn(x, W1, W2):
    """前馈网络:注意力负责「融合信息」,FFN 负责「逐 token 加工」"""
    hidden = [relu(dot(x, col)) for col in W1]      # 升维 + 非线性
    return [dot(hidden, col) for col in W2]         # 降维回 d_model


def encoder_block(X, heads_weights, W1, W2, d_k):
    """一个完整的 Encoder Block:
    x → MultiHead(x) → +x(残差) → LayerNorm → FFN → +x(残差) → LayerNorm
    """
    attn = multi_head_attention(X, heads_weights, d_k)
    # 残差连接:子层输出 + 原始输入
    res1 = [[attn[t][d] + X[t][d] for d in range(D_MODEL)] for t in range(len(X))]
    norm1 = [layer_norm(row) for row in res1]

    ffn_out = [ffn(row, W1, W2) for row in norm1]
    res2 = [[ffn_out[t][d] + norm1[t][d] for d in range(D_MODEL)] for t in range(len(X))]
    norm2 = [layer_norm(row) for row in res2]
    return norm2


# ─────────────────────────────────────────────
# 7. 格式化打印工具
# ─────────────────────────────────────────────

def fmt(v, w=6):
    return f"{v:>{w}.3f}"


def print_attention_matrix(weights, tokens):
    """把注意力权重矩阵打印成一张表:第 i 行 = token i 对各 token 的关注度"""
    print("      " + "  ".join(f"{t:>5}" for t in tokens))
    for i, row in enumerate(weights):
        print(f"{tokens[i]:>4} " + "  ".join(f"{v:>6.3f}" for v in row))
    # 行和校验
    print("  Σ   " + "  ".join(f"{sum(r):>6.2f}" for r in zip(*weights)))


# ─────────────────────────────────────────────
# 8. 主流程:组装一个小 Transformer 跑一遍
# ─────────────────────────────────────────────

def main():
    rng = make_rng(42)

    # ---- 8.1 词嵌入表 ----
    embed = make_embed(rng)

    # ---- 8.2 位置编码 ----
    pe = positional_encoding(SEQ_LEN, D_MODEL)

    # ---- 8.3 多头注意力的各组 Wq/Wk/Wv ----
    heads_weights = []
    for _ in range(N_HEADS):
        Wq = [[rng() - 0.5 for _ in range(D_K)] for _ in range(D_MODEL)]
        Wk = [[rng() - 0.5 for _ in range(D_K)] for _ in range(D_MODEL)]
        Wv = [[rng() - 0.5 for _ in range(D_K)] for _ in range(D_MODEL)]
        heads_weights.append((Wq, Wk, Wv))

    # ---- 8.4 FFN 的 W1(升维)/ W2(降维) ----
    FFN_HIDDEN = 16
    W1 = [[rng() - 0.5 for _ in range(D_MODEL)] for _ in range(FFN_HIDDEN)]
    W2 = [[rng() - 0.5 for _ in range(FFN_HIDDEN)] for _ in range(D_MODEL)]

    # ---- 8.5 准备输入句子「我 爱 猫」 ----
    sentence = "我 爱 猫"
    tokens = sentence.split(" ")
    token_ids = [TOKEN2ID[t] for t in tokens]
    print(f"输入句子: {sentence}")
    print(f"token ids: {token_ids}")
    print()

    # 词嵌入 + 位置编码(Transformer 的输入 = 嵌入 + 位置)
    X = []
    for pos, tid in enumerate(token_ids):
        x = embed_row(embed, tid)
        X.append([x[d] + pe[pos][d] for d in range(D_MODEL)])
    print("[Step 0] 输入向量 X(嵌入+位置,每个 token 8 维,只打印前 4 维):")
    for i, t in enumerate(tokens):
        print(f"   {t}: {[round(v, 3) for v in X[i][:4]]} ...")
    print()

    # ---- 8.6 单头注意力:看权重矩阵 ----
    Wq, Wk, Wv = heads_weights[0]
    _, weights = self_attention(X, Wq, Wk, Wv, D_K)
    print("[Step 1] 单头 Self-Attention 权重矩阵(第 i 行 = token i 的关注分布):")
    print_attention_matrix(weights, tokens)
    print("  说明:某格越大,说明 i 从 j 吸收的信息越多。")
    print("       (随机权重下无真实语义;真实模型训练后,这里会形成指代/语法等模式)")
    print()

    # ---- 8.7 多头注意力 ----
    multi = multi_head_attention(X, heads_weights, D_K)
    print(f"[Step 2] Multi-Head({N_HEADS} 头)输出(拼接后 {D_MODEL} 维,前 4 维):")
    for i, t in enumerate(tokens):
        print(f"   {t}: {[round(v, 3) for v in multi[i][:4]]} ...")
    print()

    # ---- 8.8 完整 Encoder Block ----
    out = encoder_block(X, heads_weights, W1, W2, D_K)
    print("[Step 3] 完整 Encoder Block 后输出(残差+LayerNorm+FFN 后):")
    for i, t in enumerate(tokens):
        print(f"   {t}: {[round(v, 3) for v in out[i][:4]]} ...")
    print()

    # ---- 8.9 一句话总结流程 ----
    print("=" * 60)
    print("流程回顾:")
    print("   嵌入+位置编码 → 多头自注意力(算词间关系) → 残差+LayerNorm")
    print("   → FFN(逐词加工) → 残差+LayerNorm → 下一层...")
    print("   → 最后一层输出喂给 Linear+Softmax → 预测下一个 token")
    print("=" * 60)


# ─────────────────────────────────────────────
# 9. 真实写法(PyTorch 对照,装了 torch 再跑)
# ─────────────────────────────────────────────

def real_world_pytorch_demo():
    """
    真实项目里你会这么写(PyTorch 官方 nn.Transformer)。
    和 ToyTransformer 对比:
        self_attention   ←→  nn.MultiheadAttention
        positional_enc   ←→  nn.TransformerEncoderLayer(内置)
        encoder_block    ←→  nn.TransformerEncoder
    内部流程完全一样,只是交给框架矩阵化 + GPU 并行。
    """
    try:
        import torch
        from torch import nn
    except ImportError:
        print("[真实写法跳过] 没装 PyTorch,`pip install torch` 后就能运行")
        return

    torch.manual_seed(42)

    # 一个单层 Transformer Encoder
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=8,        # 和 D_MODEL 对应
        nhead=2,          # 和 N_HEADS 对应
        dim_feedforward=16,  # 和 FFN_HIDDEN 对应
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    # 随机生成一个 3 个 token、每个 8 维的输入(等价于"嵌入+位置编码后")
    X = torch.randn(1, 3, 8)
    out = encoder(X)
    print("[真实写法 PyTorch] 输出 shape:", tuple(out.shape))


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
    print("下面是真实 PyTorch 写法对照:")
    print("=" * 60)
    real_world_pytorch_demo()
