import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设模型和分词器已加载
# model = AutoModelForCausalLM.from_pretrained("path/to/your/llama/model")
# tokenizer = AutoTokenizer.from_pretrained("path/to/your/llama/model")
# model.eval() # 设置为评估模式

def compute_attention_rollout(model, tokenizer, text):
    """
    计算给定文本的Attention Rollout。

    Args:
        model: 加载的语言模型 (e.g., LLaMA).
        tokenizer:对应的分词器.
        text (str): 输入文本.

    Returns:
        torch.Tensor: Attention Rollout 矩阵.
        list: 输入文本对应的 Tokens.
    """
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # 前向传播获取注意力权重
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions # attentions 是一个元组，包含每一层的注意力权重

    # attentions[layer] shape: [batch_size, num_heads, seq_len, seq_len]
    # 我们只处理 batch_size=1 的情况
    seq_len = attentions[0].shape[-1]
    
    # 初始化 Rollout 矩阵为单位矩阵
    rollout_matrix = torch.eye(seq_len, device=model.device)

    # 迭代计算 Rollout
    for layer_attention in attentions:
        # 1. 取出 batch 维度
        attn_heads = layer_attention.squeeze(0) # Shape: [num_heads, seq_len, seq_len]
        
        # 2. 平均多头注意力
        avg_heads_attn = attn_heads.mean(dim=0) # Shape: [seq_len, seq_len]
        
        # 3. (可选) 添加残差连接影响 - 简单版本：直接乘
        #    更复杂的版本会考虑残差连接的权重，例如 (attention + I) / 2
        #    这里使用基础的矩阵乘法 R_l = A_l @ R_{l-1}
        rollout_matrix = torch.matmul(avg_heads_attn, rollout_matrix)

    return rollout_matrix.cpu(), tokens # 返回 CPU 上的张量以便后续处理

def visualize_rollout(rollout_matrix, tokens, title="Attention Rollout"):
    """
    可视化 Attention Rollout 矩阵。
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(rollout_matrix.numpy(), xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title(title)
    plt.xlabel("Attended To (Key)")
    plt.ylabel("Attending From (Query)")
    plt.tight_layout()
    # plt.savefig("/home/cy131/attn-flow/attention_rollout.png") # 可以取消注释以保存图像
    plt.show()

# --- 示例用法 ---
# model_name = "meta-llama/Llama-2-7b-hf" # 替换为你实际使用的模型标识符或路径
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.eval()

# text_input = "Attention rollout helps understand transformer models."
# rollout, tokens = compute_attention_rollout(model, tokenizer, text_input)
# visualize_rollout(rollout, tokens)
# -----------------