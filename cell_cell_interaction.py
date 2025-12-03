import os
import pickle
import pandas as pd
import numpy as np
from datasets import Dataset, load_from_disk
from transformers import Trainer, TrainingArguments
from genecompass import BertForSequenceClassification, DataCollatorForCellClassification
from genecompass.utils import load_prior_embedding
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.distributed as dist
import logging
from collections import Counter
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, classification_report
import scipy.stats as stats
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_regression_metrics(pred):
    """计算回归任务的评估指标"""
    labels = pred.label_ids
    preds = pred.predictions

    # 确保preds是1D数组
    if len(preds.shape) > 1:
        preds = preds.flatten()

    # 确保labels是1D数组
    if len(labels.shape) > 1:
        labels = labels.flatten()

    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    # 计算相对误差
    abs_errors = np.abs(labels - preds)
    relative_errors = abs_errors / (np.abs(labels) + 1e-8)  # 避免除零
    mape = np.mean(relative_errors) * 100  # 平均绝对百分比误差

    # 计算相关系数
    correlation = np.corrcoef(labels, preds)[0, 1] if len(labels) > 1 else 0

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'correlation': correlation
    }


class RegressionCellInteractionDataset:
    """回归任务细胞互作数据集构建类"""

    def __init__(self, embeddings_path, gold_standard_path, dataset_path, token_dict_path):
        """
        初始化
        """
        self.embeddings_path = embeddings_path
        self.gold_standard_path = gold_standard_path
        self.dataset_path = dataset_path
        self.token_dict_path = token_dict_path

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载所有必要数据"""
        logger.info("加载基因嵌入数据...")
        with open(self.embeddings_path, 'rb') as f:
            self.gene_embeddings = pickle.load(f)

        logger.info("加载金标准标签...")
        self.gold_standard = pd.read_csv(self.gold_standard_path)

        logger.info("加载原始数据集...")
        self.original_dataset = load_from_disk(self.dataset_path)

        logger.info("加载token字典...")
        with open(self.token_dict_path, 'rb') as f:
            self.token_dictionary = pickle.load(f)

        # 获取细胞类型信息
        if 'cell_type' in self.original_dataset.column_names:
            self.cell_types = self.original_dataset['cell_type']
        else:
            self.cell_types = [f"Cell_{i}" for i in range(len(self.original_dataset))]
            logger.warning("未找到细胞类型信息，使用默认命名")

        # 创建细胞类型到索引的映射
        self.cell_to_indices = {}
        for idx, cell_type in enumerate(self.cell_types):
            if cell_type not in self.cell_to_indices:
                self.cell_to_indices[cell_type] = []
            self.cell_to_indices[cell_type].append(int(idx))

    def ensure_int_length(self, length_value):
        """确保长度值是整数类型"""
        if isinstance(length_value, (list, np.ndarray)):
            if len(length_value) > 0:
                return int(length_value[0])
            else:
                return 0
        elif isinstance(length_value, (int, np.integer)):
            return int(length_value)
        else:
            try:
                return int(length_value)
            except (ValueError, TypeError):
                logger.warning(f"无法将长度值 {length_value} 转换为整数，使用默认值0")
                return 0

    def _build_cell_pair_sequence(self, dataset, sender_idx, receiver_idx, max_sequence_length):
        """构建细胞对序列 - 修复空序列问题"""
        try:
            # 获取细胞数据
            sender_data = dataset[sender_idx]
            receiver_data = dataset[receiver_idx]

            # 获取特殊token
            cls_token = self.token_dictionary.get("<cls>", 1)
            sep_token = self.token_dictionary.get("<sep>", 2)
            pad_token = self.token_dictionary.get("<pad>", 0)

            # 安全获取序列数据
            sender_input_ids = sender_data.get('input_ids', [])
            sender_values = sender_data.get('values', [])
            sender_length_raw = sender_data.get('length', 0)

            receiver_input_ids = receiver_data.get('input_ids', [])
            receiver_values = receiver_data.get('values', [])
            receiver_length_raw = receiver_data.get('length', 0)

            # 确保序列数据有效
            if (not sender_input_ids or not receiver_input_ids or
                    not sender_values or not receiver_values):
                return None

            # 确保长度值是整数
            sender_length = self.ensure_int_length(sender_length_raw)
            receiver_length = self.ensure_int_length(receiver_length_raw)

            # 检查长度有效性
            if sender_length <= 0 or receiver_length <= 0:
                return None

            # 计算可用长度
            available_length = max_sequence_length - 3
            total_length = sender_length + receiver_length

            if total_length > available_length:
                # 按比例分配长度
                sender_ratio = sender_length / total_length
                sender_alloc = max(1, int(available_length * sender_ratio))
                receiver_alloc = max(1, available_length - sender_alloc)
            else:
                sender_alloc = sender_length
                receiver_alloc = receiver_length

            # 确保分配长度有效
            sender_alloc = max(1, min(sender_alloc, len(sender_input_ids)))
            receiver_alloc = max(1, min(receiver_alloc, len(receiver_input_ids)))

            # 截断序列
            sender_input_ids_trunc = sender_input_ids[:sender_alloc]
            sender_values_trunc = sender_values[:sender_alloc]
            receiver_input_ids_trunc = receiver_input_ids[:receiver_alloc]
            receiver_values_trunc = receiver_values[:receiver_alloc]

            # 检查截断后是否为空
            if not sender_input_ids_trunc or not receiver_input_ids_trunc:
                return None

            # 构建细胞对序列
            pair_input_ids = [cls_token]
            pair_input_ids.extend(sender_input_ids_trunc)
            pair_input_ids.append(sep_token)
            pair_input_ids.extend(receiver_input_ids_trunc)
            pair_input_ids.append(sep_token)

            # 构建值序列
            pair_values = [0.0]
            pair_values.extend(sender_values_trunc)
            pair_values.append(0.0)
            pair_values.extend(receiver_values_trunc)
            pair_values.append(0.0)

            # 填充到固定长度
            current_length = len(pair_input_ids)
            if current_length < max_sequence_length:
                pad_length = max_sequence_length - current_length
                pair_input_ids.extend([pad_token] * pad_length)
                pair_values.extend([0.0] * pad_length)
            else:
                pair_input_ids = pair_input_ids[:max_sequence_length]
                pair_values = pair_values[:max_sequence_length]

            # 最终验证
            if len(pair_input_ids) != max_sequence_length or len(pair_values) != max_sequence_length:
                return None

            sequence_length = len(sender_input_ids_trunc) + len(receiver_input_ids_trunc) + 3

            return {
                'input_ids': pair_input_ids,
                'values': pair_values,
                'length': sequence_length
            }

        except Exception as e:
            logger.error(f"构建细胞对序列失败: {str(e)}")
            return None

    def create_cell_pair_sequences(self, max_sequence_length=2048, balance_dataset=True):
        """
        创建细胞对序列数据 - 回归任务版本
        """
        logger.info("创建细胞对序列（回归任务）...")

        sequences = []
        labels = []
        cell_pairs = []

        # 查找连续分数列 - 回归任务使用连续分数
        score_column = None
        for col in ['Consensus_Score', 'Interaction_Score', 'Score', 'score', 'value']:
            if col in self.gold_standard.columns:
                score_column = col
                logger.info(f"使用连续分数列: {score_column}")
                break

        if score_column is None:
            logger.error("未找到连续分数列，请检查金标准文件格式")
            # 尝试查看所有列
            logger.error(f"可用列: {list(self.gold_standard.columns)}")
            return sequences, labels, cell_pairs

        # 适配发送者和接收者列名
        sender_column = None
        receiver_column = None

        for col in ['Sender', 'sender', 'source', 'from']:
            if col in self.gold_standard.columns:
                sender_column = col
                break

        for col in ['Receiver', 'receiver', 'target', 'to']:
            if col in self.gold_standard.columns:
                receiver_column = col
                break

        if sender_column is None or receiver_column is None:
            logger.error("未找到发送者或接收者列")
            logger.error(f"可用列: {list(self.gold_standard.columns)}")
            return sequences, labels, cell_pairs

        logger.info(f"使用发送者列: {sender_column}, 接收者列: {receiver_column}")

        # 过滤无效分数
        valid_data = self.gold_standard.dropna(subset=[score_column])
        valid_data = valid_data[valid_data[score_column] >= 0]  # 确保分数非负

        logger.info(f"有效数据数量: {len(valid_data)}")
        logger.info(f"分数统计 - 最小值: {valid_data[score_column].min():.4f}, "
                    f"最大值: {valid_data[score_column].max():.4f}, "
                    f"平均值: {valid_data[score_column].mean():.4f}")

        valid_sequences_count = 0
        invalid_sequences_count = 0

        for _, row in tqdm(valid_data.iterrows(), total=len(valid_data), desc="处理细胞对"):
            sender_type = row[sender_column]
            receiver_type = row[receiver_column]
            score = float(row[score_column])

            if sender_type in self.cell_to_indices and receiver_type in self.cell_to_indices:
                sender_idx = int(np.random.choice(self.cell_to_indices[sender_type]))
                receiver_idx = int(np.random.choice(self.cell_to_indices[receiver_type]))

                # 构建序列
                sequence = self._build_cell_pair_sequence(
                    self.original_dataset, sender_idx, receiver_idx, max_sequence_length
                )
                if sequence is not None:
                    sequences.append(sequence)
                    labels.append(score)
                    cell_pairs.append(f"{sender_type}_{receiver_type}")
                    valid_sequences_count += 1
                else:
                    invalid_sequences_count += 1
            else:
                invalid_sequences_count += 1

        if invalid_sequences_count > 0:
            logger.warning(f"跳过 {invalid_sequences_count} 个无效序列")

        logger.info(f"创建了 {len(sequences)} 个有效的细胞对序列")
        logger.info(f"分数范围: {min(labels):.4f} - {max(labels):.4f}, 平均值: {np.mean(labels):.4f}")
        return sequences, labels, cell_pairs

    def create_huggingface_dataset(self, sequences, labels, test_size=0.2, validation_size=0.1):
        """创建HuggingFace数据集"""
        logger.info("创建HuggingFace数据集（回归任务）...")

        if len(sequences) == 0:
            logger.error("没有有效的序列数据")
            return None, None, None

        # 添加物种信息（默认为0）
        species = [0] * len(sequences)

        dataset_dict = {
            'input_ids': [seq['input_ids'] for seq in sequences],
            'values': [seq['values'] for seq in sequences],
            'length': [seq['length'] for seq in sequences],
            'species': species,
            'label': labels  # 回归任务使用连续分数作为标签
        }

        # 创建数据集
        dataset = Dataset.from_dict(dataset_dict)

        # 划分数据集
        if len(dataset) > 1:
            train_val_test = dataset.train_test_split(test_size=test_size + validation_size, seed=42)

            if len(train_val_test['test']) > 1:
                # 计算验证集和测试集的比例
                val_ratio = validation_size / (test_size + validation_size)
                val_test = train_val_test['test'].train_test_split(
                    test_size=val_ratio, seed=42
                )
                train_dataset = train_val_test['train']
                val_dataset = val_test['train']
                test_dataset = val_test['test']
            else:
                train_dataset = train_val_test['train']
                val_dataset = train_val_test['test']
                test_dataset = train_val_test['test']
        else:
            logger.warning("数据量过少，使用全部数据作为训练集")
            train_dataset = dataset
            val_dataset = dataset
            test_dataset = dataset

        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        logger.info(f"测试集大小: {len(test_dataset)}")
        logger.info(f"训练集分数范围: {min(train_dataset['label']):.4f} - {max(train_dataset['label']):.4f}")

        return train_dataset, val_dataset, test_dataset


def setup_multigpu_training():
    """设置多GPU训练环境"""
    if torch.cuda.device_count() > 1:
        logger.info(f"检测到 {torch.cuda.device_count()} 个GPU可用")

        if not dist.is_initialized():
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                rank = int(os.environ['RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
                dist.init_process_group(backend='nccl', init_method='env://')
                logger.info(f"初始化分布式训练: rank={rank}, world_size={world_size}")
                return True
            else:
                logger.info("单机多GPU模式，但未初始化分布式训练")
                return False
        return True
    else:
        logger.info("单GPU训练模式")
        return False


def fine_tune_regression_model(config):
    """
    微调细胞互作回归模型
    """
    logger.info("开始细胞互作关系回归模型微调...")

    # 设置多GPU训练
    is_multigpu = setup_multigpu_training()

    # 获取当前进程的rank
    if is_multigpu and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"当前进程rank: {rank}, 总进程数: {world_size}")
    else:
        rank = 0
        world_size = 1

    # 只有主进程创建输出目录
    if rank == 0:
        os.makedirs(config['output_dir'], exist_ok=True)
        logger.info(f"创建输出目录: {config['output_dir']}")

    # 1. 创建数据集
    if rank == 0:
        logger.info("主进程创建数据集...")
        try:
            dataset_builder = RegressionCellInteractionDataset(
                embeddings_path=config['embeddings_path'],
                gold_standard_path=config['gold_standard_path'],
                dataset_path=config['dataset_path'],
                token_dict_path=config['token_dict_path']
            )

            # 创建细胞对序列
            sequences, labels, cell_pairs = dataset_builder.create_cell_pair_sequences(
                max_sequence_length=config.get('max_sequence_length', 2048),
                balance_dataset=config.get('balance_dataset', False)  # 回归任务通常不需要平衡
            )

            # 检查数据集是否有效
            if len(sequences) == 0:
                logger.error("创建的数据集为空，无法进行训练")
                return None, None, None

            # 创建HuggingFace数据集
            train_dataset, val_dataset, test_dataset = dataset_builder.create_huggingface_dataset(
                sequences, labels,
                test_size=config.get('test_size', 0.2),
                validation_size=config.get('validation_size', 0.1)
            )

            # 保存数据集
            dataset_save_path = os.path.join(config['output_dir'], "temp_dataset")
            os.makedirs(dataset_save_path, exist_ok=True)

            train_dataset.save_to_disk(os.path.join(dataset_save_path, "train"))
            val_dataset.save_to_disk(os.path.join(dataset_save_path, "val"))
            test_dataset.save_to_disk(os.path.join(dataset_save_path, "test"))
            logger.info("数据集已保存")
        except Exception as e:
            logger.error(f"创建数据集失败: {str(e)}")
            return None, None, None
    else:
        dataset_save_path = os.path.join(config['output_dir'], "temp_dataset")
        logger.info(f"进程 {rank} 等待数据集...")

    # 同步所有进程
    if is_multigpu and dist.is_initialized():
        dist.barrier()

    # 所有进程加载数据集
    if rank != 0 or (rank == 0 and 'train_dataset' not in locals()):
        logger.info(f"进程 {rank} 加载数据集...")
        try:
            train_dataset = load_from_disk(os.path.join(dataset_save_path, "train"))
            val_dataset = load_from_disk(os.path.join(dataset_save_path, "val"))
            test_dataset = load_from_disk(os.path.join(dataset_save_path, "test"))
            logger.info(f"进程 {rank} 数据集加载完成")
        except Exception as e:
            logger.error(f"进程 {rank} 数据集加载失败: {str(e)}")
            return None, None, None

    # 2. 加载先验知识
    logger.info("加载先验知识...")
    knowledges = {}
    try:
        out = load_prior_embedding(token_dictionary_or_path=config['token_dict_path'])
        knowledges['promoter'] = out[0] if len(out) > 0 else None
        knowledges['co_exp'] = out[1] if len(out) > 1 else None
        knowledges['gene_family'] = out[2] if len(out) > 2 else None
        knowledges['peca_grn'] = out[3] if len(out) > 3 else None
        knowledges['homologous_gene_human2mouse'] = out[4] if len(out) > 4 else None
        logger.info("先验知识加载成功")
    except Exception as e:
        logger.warning(f"加载先验知识失败: {str(e)}")
        knowledges = {
            'promoter': None, 'co_exp': None, 'gene_family': None,
            'peca_grn': None, 'homologous_gene_human2mouse': None
        }

    # 3. 加载预训练模型 - 回归任务使用num_labels=1
    logger.info("加载预训练模型（回归任务）...")
    try:
        # 回归任务，设置num_labels=1
        model = BertForSequenceClassification.from_pretrained(
            config['model_path'],
            num_labels=1,  # 回归任务
            output_attentions=False,
            output_hidden_states=False,
            knowledges=knowledges,
        )
        logger.info("预训练模型加载成功（回归任务）")
    except Exception as e:
        logger.error(f"加载预训练模型失败: {str(e)}")
        return None, None, None

    # 4. 冻结部分层（可选）
    if config.get('freeze_layers', 0) > 0:
        logger.info(f"冻结前 {config['freeze_layers']} 层")
        freeze_layers = config['freeze_layers']
        if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
            modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

    # 5. 设置训练参数 - 回归任务
    per_device_batch_size = config.get('batch_size', 4)

    # 根据数据集大小调整批次大小
    if len(train_dataset) > 100:
        per_device_batch_size = min(per_device_batch_size, 8)
    else:
        per_device_batch_size = min(per_device_batch_size, 2)

    if is_multigpu and world_size > 1:
        effective_batch_size = per_device_batch_size * world_size
        logger.info(f"多GPU训练: 每设备批次大小={per_device_batch_size}, 有效批次大小={effective_batch_size}")
    else:
        effective_batch_size = per_device_batch_size
        logger.info(f"单GPU训练: 批次大小={effective_batch_size}")

    # 调整训练轮数基于数据集大小
    num_epochs = config.get('num_epochs', 40)
    if len(train_dataset) > 100:
        num_epochs = min(num_epochs, 30)

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=config.get('learning_rate', 5e-5),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(config['output_dir'], "logs"),
        logging_steps=10,
        disable_tqdm=False,
        lr_scheduler_type="linear",
        warmup_steps=config.get('warmup_steps', 100),
        weight_decay=config.get('weight_decay', 0.001),
        load_best_model_at_end=True,
        metric_for_best_model=config.get('metric_for_best_model', 'rmse'),
        greater_is_better=False,
        dataloader_num_workers=min(2, os.cpu_count() // 2),
        dataloader_pin_memory=True,
        fp16=config.get('fp16', True),
        local_rank=rank,
        ddp_find_unused_parameters=False,
        report_to=[],
        remove_unused_columns=True  # 关键修复：自动移除不使用的列，防止length参数错误
    )

    # 6. 创建训练器 - 使用回归评估指标
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForCellClassification(),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_regression_metrics
        )
    except Exception as e:
        logger.error(f"创建训练器失败: {str(e)}")
        return None, None, None

    # 7. 开始训练
    logger.info("开始回归模型训练...")
    try:
        train_result = trainer.train()
        logger.info("回归模型训练完成")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        return None, None, None

    # 8. 保存模型和评估结果
    if rank == 0:
        logger.info("保存回归模型和评估结果...")

        try:
            # 在测试集上评估
            test_predictions = trainer.predict(test_dataset)
            val_predictions = trainer.predict(val_dataset)

            # 保存评估指标
            with open(os.path.join(config['output_dir'], "test_metrics.json"), 'w') as f:
                json.dump(test_predictions.metrics, f, indent=2)

            # 保存模型
            trainer.save_model(config['output_dir'])

            # 保存训练历史
            training_history = {
                'train_loss': [log for log in trainer.state.log_history if 'loss' in log],
                'eval_metrics': [log for log in trainer.state.log_history if 'eval_loss' in log]
            }
            with open(os.path.join(config['output_dir'], "training_history.pkl"), 'wb') as f:
                pickle.dump(training_history, f)

            # 保存预测结果
            predictions_df = pd.DataFrame({
                'true_labels': test_predictions.label_ids.flatten(),
                'predicted_scores': test_predictions.predictions.flatten()
            })
            predictions_df.to_csv(os.path.join(config['output_dir'], "test_predictions.csv"), index=False)

            # 清理临时数据集
            if os.path.exists(dataset_save_path):
                import shutil
                shutil.rmtree(dataset_save_path)
                logger.info("清理临时数据集")

            logger.info(f"回归模型训练完成！模型和结果已保存到: {config['output_dir']}")
        except Exception as e:
            logger.error(f"保存结果时出现错误: {str(e)}")
            test_predictions = None
    else:
        test_predictions = None

    # 同步所有进程
    if is_multigpu and dist.is_initialized():
        dist.barrier()

    return trainer, test_predictions, val_predictions


class RegressionCellInteractionPredictor:
    """回归任务细胞互作预测器"""

    def __init__(self, model_path, token_dict_path):
        """
        初始化预测器
        """
        self.model_path = model_path
        self.token_dict_path = token_dict_path
        self.model = None
        self.token_dictionary = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型和token字典
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        """加载模型和token字典"""
        logger.info("加载回归模型和token字典...")

        # 加载token字典
        with open(self.token_dict_path, 'rb') as f:
            self.token_dictionary = pickle.load(f)

        # 加载先验知识
        knowledges = {}
        try:
            out = load_prior_embedding(token_dictionary_or_path=self.token_dict_path)
            knowledges['promoter'] = out[0] if len(out) > 0 else None
            knowledges['co_exp'] = out[1] if len(out) > 1 else None
            knowledges['gene_family'] = out[2] if len(out) > 2 else None
            knowledges['peca_grn'] = out[3] if len(out) > 3 else None
            knowledges['homologous_gene_human2mouse'] = out[4] if len(out) > 4 else None
        except Exception as e:
            logger.warning(f"加载先验知识失败: {str(e)}")
            knowledges = {
                'promoter': None, 'co_exp': None, 'gene_family': None,
                'peca_grn': None, 'homologous_gene_human2mouse': None
            }

        # 加载模型 - 回归任务
        try:
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path,
                knowledges=knowledges,
            )
            logger.info("回归模型加载成功")
        except Exception as e:
            logger.error(f"回归模型加载失败: {str(e)}")
            raise

        self.model.eval()
        self.model = self.model.to(self.device)
        logger.info("回归模型加载完成")

    def _create_cell_index_mapping(self, dataset):
        """创建细胞类型到索引的映射"""
        cell_to_indices = {}
        if 'cell_type' in dataset.column_names:
            cell_types = dataset['cell_type']
        else:
            cell_types = [f"Cell_{i}" for i in range(len(dataset))]
        for idx, cell_type in enumerate(cell_types):
            if cell_type not in cell_to_indices:
                cell_to_indices[cell_type] = []
            cell_to_indices[cell_type].append(idx)
        return cell_to_indices

    def _build_cell_pair_sequence(self, dataset, sender, receiver, max_sequence_length):
        """构建细胞对序列 - 完整实现"""
        try:
            # 创建细胞类型到索引的映射
            cell_to_indices = self._create_cell_index_mapping(dataset)

            # 检查细胞类型是否存在
            if sender not in cell_to_indices or receiver not in cell_to_indices:
                logger.warning(f"细胞类型 {sender} 或 {receiver} 不在数据集中")
                return None

            # 随机选择索引
            sender_idx = int(np.random.choice(cell_to_indices[sender]))
            receiver_idx = int(np.random.choice(cell_to_indices[receiver]))

            # 获取细胞数据
            sender_data = dataset[sender_idx]
            receiver_data = dataset[receiver_idx]

            # 获取特殊token
            cls_token = self.token_dictionary.get("<cls>", 1)
            sep_token = self.token_dictionary.get("<sep>", 2)
            pad_token = self.token_dictionary.get("<pad>", 0)

            # 安全获取序列数据
            sender_input_ids = sender_data.get('input_ids', [])
            sender_values = sender_data.get('values', [])
            sender_length = len(sender_input_ids)

            receiver_input_ids = receiver_data.get('input_ids', [])
            receiver_values = receiver_data.get('values', [])
            receiver_length = len(receiver_input_ids)

            # 确保序列数据有效
            if (not sender_input_ids or not receiver_input_ids or
                    not sender_values or not receiver_values):
                return None

            # 计算可用长度
            available_length = max_sequence_length - 3
            total_length = sender_length + receiver_length

            if total_length > available_length:
                # 按比例分配长度
                sender_ratio = sender_length / total_length
                sender_alloc = max(1, int(available_length * sender_ratio))
                receiver_alloc = max(1, available_length - sender_alloc)
            else:
                sender_alloc = sender_length
                receiver_alloc = receiver_length

            # 确保分配长度有效
            sender_alloc = max(1, min(sender_alloc, len(sender_input_ids)))
            receiver_alloc = max(1, min(receiver_alloc, len(receiver_input_ids)))

            # 截断序列
            sender_input_ids_trunc = sender_input_ids[:sender_alloc]
            sender_values_trunc = sender_values[:sender_alloc]
            receiver_input_ids_trunc = receiver_input_ids[:receiver_alloc]
            receiver_values_trunc = receiver_values[:receiver_alloc]

            # 检查截断后是否为空
            if not sender_input_ids_trunc or not receiver_input_ids_trunc:
                return None

            # 构建细胞对序列
            pair_input_ids = [cls_token]
            pair_input_ids.extend(sender_input_ids_trunc)
            pair_input_ids.append(sep_token)
            pair_input_ids.extend(receiver_input_ids_trunc)
            pair_input_ids.append(sep_token)

            # 构建值序列
            pair_values = [0.0]
            pair_values.extend(sender_values_trunc)
            pair_values.append(0.0)
            pair_values.extend(receiver_values_trunc)
            pair_values.append(0.0)

            # 填充到固定长度
            current_length = len(pair_input_ids)
            if current_length < max_sequence_length:
                pad_length = max_sequence_length - current_length
                pair_input_ids.extend([pad_token] * pad_length)
                pair_values.extend([0.0] * pad_length)
            else:
                pair_input_ids = pair_input_ids[:max_sequence_length]
                pair_values = pair_values[:max_sequence_length]

            # 最终验证
            if len(pair_input_ids) != max_sequence_length or len(pair_values) != max_sequence_length:
                return None

            sequence_length = len(sender_input_ids_trunc) + len(receiver_input_ids_trunc) + 3

            return {
                'input_ids': pair_input_ids,
                'values': pair_values,
                'length': sequence_length
            }

        except Exception as e:
            logger.error(f"构建细胞对序列失败: {str(e)}")
            return None

    def predict_interaction_scores(self, dataset, cell_types, max_sequence_length=2048, batch_size=8):
        """
        预测细胞互作分数
        """
        logger.info("开始预测细胞互作分数（回归任务）...")

        # 生成所有可能的细胞对
        cell_pairs = []
        for i, sender in enumerate(cell_types):
            for j, receiver in enumerate(cell_types):
                if sender != receiver:  # 排除自互作
                    cell_pairs.append((sender, receiver))

        logger.info(f"需要预测 {len(cell_pairs)} 个细胞对")

        predictions = []

        with torch.no_grad():
            for i in tqdm(range(0, len(cell_pairs), batch_size), desc="预测"):
                batch_predictions = self._predict_batch(
                    dataset, cell_pairs[i:i + batch_size], max_sequence_length
                )
                predictions.extend(batch_predictions)

        logger.info(f"完成 {len(predictions)} 个预测")
        return predictions

    def _predict_batch(self, dataset, cell_pairs, max_sequence_length):
        """批量预测"""
        batch_predictions = []

        for sender, receiver in cell_pairs:
            try:
                # 构建序列
                sequence = self._build_cell_pair_sequence(dataset, sender, receiver, max_sequence_length)
                if sequence is None:
                    continue

                # 准备输入
                input_ids = torch.tensor([sequence['input_ids']]).long().to(self.device)
                values = torch.tensor([sequence['values']]).float().to(self.device)

                # 预测 - 回归任务直接输出分数
                outputs = self.model(input_ids=input_ids, values=values)
                score = outputs.logits.item()  # 回归任务直接取标量值

                batch_predictions.append({
                    'sender': sender,
                    'receiver': receiver,
                    'predicted_score': score,
                    'confidence': 1.0  # 回归任务没有置信度，设为1.0
                })
            except Exception as e:
                logger.warning(f"预测细胞对 {sender}-{receiver} 失败: {str(e)}")
                continue

        return batch_predictions

    def create_interaction_matrix(self, predictions, cell_types):
        """
        创建细胞互作评分矩阵
        """
        logger.info("创建细胞互作评分矩阵...")

        # 创建空的DataFrame
        interaction_matrix = pd.DataFrame(
            0.0, index=cell_types, columns=cell_types
        )

        # 填充预测结果
        for pred in predictions:
            sender = pred['sender']
            receiver = pred['receiver']
            score = pred['predicted_score']
            if sender in cell_types and receiver in cell_types:
                interaction_matrix.loc[sender, receiver] = score

        return interaction_matrix

    def save_results(self, interaction_matrix, predictions, output_dir):
        """保存预测结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存互作矩阵
        matrix_path = os.path.join(output_dir, 'interaction_score_matrix.csv')
        interaction_matrix.to_csv(matrix_path)
        logger.info(f"互作评分矩阵已保存: {matrix_path}")

        # 保存详细预测结果
        predictions_df = pd.DataFrame(predictions)
        predictions_path = os.path.join(output_dir, 'detailed_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"详细预测结果已保存: {predictions_path}")

        # 生成统计分析
        stats = self._generate_statistics(interaction_matrix, predictions)
        stats_path = os.path.join(output_dir, 'statistical_analysis.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"统计分析已保存: {stats_path}")

        # 生成可视化
        self._generate_visualizations(interaction_matrix, output_dir)
        logger.info("可视化结果已生成")

    def _generate_statistics(self, interaction_matrix, predictions):
        """生成统计分析"""
        scores = [pred['predicted_score'] for pred in predictions]

        stats = {
            "basic_statistics": {
                "total_cell_types": interaction_matrix.shape[0],
                "total_predicted_interactions": len(predictions),
                "score_range": {
                    "min": float(min(scores)),
                    "max": float(max(scores)),
                    "mean": float(np.mean(scores)),
                    "median": float(np.median(scores)),
                    "std": float(np.std(scores))
                },
                "high_score_interactions": len([s for s in scores if s > np.median(scores)]),
                "low_score_interactions": len([s for s in scores if s <= np.median(scores)])
            },
            "top_interactions": sorted(predictions, key=lambda x: x['predicted_score'], reverse=True)[:20]
        }
        if not predictions:  # 检查predictions是否为空
            logger.warning("预测结果为空，返回默认统计信息")
            return {
                "basic_statistics": {
                    "total_cell_types": interaction_matrix.shape[0],
                    "total_predicted_interactions": 0,
                    "score_range": {
                        "min": 0.0,
                        "max": 0.0,
                        "mean": 0.0,
                        "median": 0.0,
                        "std": 0.0
                    },
                    "high_score_interactions": 0,
                    "low_score_interactions": 0
                },
                "top_interactions": []
            }

        return stats

    def _generate_visualizations(self, interaction_matrix, output_dir):
        """生成可视化结果"""
        try:
            plt.style.use('default')

            # 1. 互作评分矩阵热图
            plt.figure(figsize=(12, 10))
            sns.heatmap(interaction_matrix, annot=True, fmt='.3f', cmap='Reds',
                        cbar_kws={'label': 'Interaction Score'})
            plt.title('Cell Interaction Score Matrix (Regression)')
            plt.tight_layout()
            heatmap_path = os.path.join(output_dir, 'interaction_score_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 评分分布直方图
            plt.figure(figsize=(10, 6))
            scores = interaction_matrix.values.flatten()
            scores = scores[scores > 0]  # 只显示正分数
            plt.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Interaction Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Interaction Scores (Regression)')
            plt.grid(True, alpha=0.3)
            hist_path = os.path.join(output_dir, 'score_distribution.png')
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("可视化结果已生成")
        except Exception as e:
            logger.warning(f"生成可视化结果失败: {str(e)}")


# 主函数
def main():
    """主执行函数"""
    # 配置参数 - 回归任务版本
    config = {
        'embeddings_path': '/mnt/data_sdb/wangx/data/SingleCell/gene_embeddings/TabulaSapiens/tabula_sapiens_liver/gene_embeddings_liver.pickle',
        'gold_standard_path': '/mnt/data_sdb/wangx/GeneCompass/cell_interaction_gold_standard/liver/complete_labeled_interactions.csv',
        'dataset_path': '/mnt/data_sdb/wangx/data/SingleCell/normalized_data/TabulaSapiens/tabula_sapiens_liver/tabula_sapiens_liver/',
        'token_dict_path': './prior_knowledge/human_mouse_tokens.pickle',
        'model_path': './pretrained_models/GeneCompass_Base',
        'output_dir': '/mnt/data_sdb/wangx/GeneCompass/cell_interaction_outputs/results_20251030/',
        'max_sequence_length': 2048,
        'batch_size': 2,
        'num_epochs': 30,
        'learning_rate': 5e-5,
        'warmup_steps': 100,
        'weight_decay': 0.001,
        'test_size': 0.2,
        'validation_size': 0.1,
        'balance_dataset': False,
        'metric_for_best_model': 'rmse'  # 回归任务使用RMSE作为最佳模型指标
    }

    # 阶段1: 微调回归模型
    logger.info("=== 阶段1: 细胞互作回归模型微调 ===")
    trainer, test_predictions, val_predictions = fine_tune_regression_model(config)

    # 打印评估结果
    if trainer is not None and test_predictions is not None:
        logger.info("=== 回归模型评估结果 ===")
        for metric, value in test_predictions.metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    # 阶段2: 使用微调后的回归模型进行预测
    if trainer is not None:
        logger.info("=== 阶段2: 细胞互作评分预测 ===")

        # 创建预测器并进行预测
        predictor = RegressionCellInteractionPredictor(
            model_path=config['output_dir'],
            token_dict_path=config['token_dict_path']
        )

        # 加载数据集以获取细胞类型信息
        dataset = load_from_disk(config['dataset_path'])
        if 'cell_type' in dataset.column_names:
            cell_types = sorted(list(set(dataset['cell_type'])))
        else:
            cell_types = [f"Cell_{i}" for i in range(len(dataset))]

        # 进行预测
        predictions = predictor.predict_interaction_scores(
            dataset=dataset,
            cell_types=cell_types,
            max_sequence_length=config['max_sequence_length'],
            batch_size=config['batch_size']
        )

        # 创建互作评分矩阵
        interaction_matrix = predictor.create_interaction_matrix(predictions, cell_types)

        # 保存结果
        output_dir = os.path.join(config['output_dir'], 'predictions')
        predictor.save_results(interaction_matrix, predictions, output_dir)

        logger.info("=== 回归预测完成 ===")
        logger.info(f"结果保存在: {output_dir}")

    logger.info("=== 处理完成 ===")


if __name__ == "__main__":
    main()
