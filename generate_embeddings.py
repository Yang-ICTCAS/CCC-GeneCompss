from datasets import load_from_disk
import torch
from tqdm import tqdm
import pickle
import os
import gc
from genecompass.modeling_bert import BertForMaskedLM

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
token_dict_path = './prior_knowledge/human_mouse_tokens.pickle'
dataset_path = '/mnt/data_sdb/wangx/data/SingleCell/normalized_data/TabulaSapiens/tabula_sapiens_liver/'

checkpoint_path = "./pretrained_models/GeneCompass_Base"

with open(token_dict_path, "rb") as fp:
    token_dictionary = pickle.load(fp)

# 加载知识嵌入
knowledges = dict()
from genecompass.utils import load_prior_embedding

out = load_prior_embedding(token_dictionary_or_path=token_dict_path)

knowledges['promoter'] = out[0]
knowledges['co_exp'] = out[1]
knowledges['gene_family'] = out[2]
knowledges['peca_grn'] = out[3]
knowledges['homologous_gene_human2mouse'] = out[4]

# 加载数据集和模型
dataset = load_from_disk(dataset_path)
model = BertForMaskedLM.from_pretrained(
    checkpoint_path,
    knowledges=knowledges,
).to("cuda")
model.eval()

# 显存优化配置
torch.backends.cudnn.benchmark = True  # 加速卷积运算
torch.cuda.empty_cache()  # 清空缓存


# 批量处理函数 - 优化显存使用
def process_batch_optimized(batch_indices, dataset, model):
    with torch.no_grad():
        # 一次性获取所有数据
        start_idx, end_idx = batch_indices
        input_id = torch.tensor(dataset['input_ids'][start_idx:end_idx]).cuda()
        values = torch.tensor(dataset['values'][start_idx:end_idx]).cuda()
        species = torch.tensor(dataset['species'][start_idx:end_idx]).cuda()

        # 前向传播
        emb = model.bert.forward(input_ids=input_id, values=values, species=species)[0]
        emb = emb[:, 1:, :]  # 去除第一个token

        # 立即移动到CPU并转换为numpy以释放显存
        emb_cpu = emb.cpu().numpy()

        # 清理GPU显存
        del input_id, values, species, emb
        torch.cuda.empty_cache()

        return emb_cpu


# 主处理循环
batchsize = 128
total_length = len(dataset)
iters = total_length // batchsize

print(f"开始处理，总样本数: {total_length}, 批次大小: {batchsize}, 总迭代次数: {iters + 1}")

# 使用列表存储CPU上的结果
emb_list = []

with torch.no_grad():
    for i in tqdm(range(iters + 1), desc="处理批次"):
        try:
            # 计算当前批次索引
            start_idx = i * batchsize
            if i != iters:
                end_idx = (i + 1) * batchsize
            else:
                end_idx = total_length

            # 处理当前批次
            emb_batch = process_batch_optimized((start_idx, end_idx), dataset, model)
            emb_list.append(emb_batch)

            # 定期清理（每10个批次）
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"批次 {i} 显存不足，尝试减小批次大小...")
                # 可以在这里添加批次大小减半的逻辑
                raise e
            else:
                raise e

# 保存结果
output_path = os.path.join(dataset_path, 'gene_embeddings.pickle')
print(f"保存结果到: {output_path}")

# 拼接所有结果
import numpy as np

emb_numpy = np.concatenate(emb_list, axis=0)

with open(output_path, 'wb') as f:
    pickle.dump(emb_numpy, f)

print("处理完成！")


