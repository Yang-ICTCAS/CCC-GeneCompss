import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import re
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_cellchat_results(cellchat_dir):
    """
    加载CellChat结果文件
    """
    logger.info("加载CellChat结果...")
    results = {}

    # 1. 加载互作强度矩阵
    interaction_matrix_path = os.path.join(cellchat_dir, "cell_interaction_strength_matrix.csv")
    if not os.path.exists(interaction_matrix_path):
        # 尝试其他可能的文件名
        alt_names = ['cell_interaction_matrix.csv', 'interaction_matrix.csv', 'interaction_strength.csv']
        for alt in alt_names:
            alt_path = os.path.join(cellchat_dir, alt)
            if os.path.exists(alt_path):
                interaction_matrix_path = alt_path
                break
        else:
            raise FileNotFoundError(f"CellChat互作矩阵文件不存在: {interaction_matrix_path}")

    logger.info(f"加载互作矩阵: {interaction_matrix_path}")
    try:
        cellchat_matrix = pd.read_csv(interaction_matrix_path)

        # 检查列名
        if 'Sender' not in cellchat_matrix.columns:
            # 尝试查找可能的列名
            sender_cols = [col for col in cellchat_matrix.columns if 'sender' in col.lower() or 'source' in col.lower()]
            if sender_cols:
                cellchat_matrix.rename(columns={sender_cols[0]: 'Sender'}, inplace=True)
            else:
                # 如果第一列不是发送者，可能是索引列
                if cellchat_matrix.columns[0] != 'Sender':
                    cellchat_matrix.rename(columns={cellchat_matrix.columns[0]: 'Sender'}, inplace=True)

        # 转换成长格式
        receiver_cols = [col for col in cellchat_matrix.columns if col != 'Sender']
        cellchat_matrix = cellchat_matrix.melt(
            id_vars='Sender',
            value_vars=receiver_cols,
            var_name='Receiver',
            value_name='CellChat_Score'
        )

        # 确保分数是数值
        cellchat_matrix['CellChat_Score'] = pd.to_numeric(
            cellchat_matrix['CellChat_Score'], errors='coerce'
        ).fillna(0)

        results['matrix'] = cellchat_matrix
    except Exception as e:
        logger.error(f"加载互作矩阵失败: {str(e)}")
        raise

    # 2. 加载细胞通讯结果
    communication_path = os.path.join(cellchat_dir, "cellchat_communication.csv")
    if not os.path.exists(communication_path):
        # 尝试其他可能的文件名
        alt_names = ['communication.csv', 'cell_communication.csv', 'interaction_results.csv']
        for alt in alt_names:
            alt_path = os.path.join(cellchat_dir, alt)
            if os.path.exists(alt_path):
                communication_path = alt_path
                break
        else:
            logger.warning(f"CellChat通讯文件不存在: {communication_path}")
            results['communication'] = pd.DataFrame()
            results['pathways'] = pd.DataFrame()
            return results

    logger.info(f"加载通讯文件: {communication_path}")
    try:
        cellchat_comm = pd.read_csv(communication_path)

        # 检查并重命名列
        col_mapping = {
            'source': ['source', 'sender', 'cell_source', 'from'],
            'target': ['target', 'receiver', 'cell_target', 'to'],
            'interaction_name': ['interaction_name', 'interaction', 'pathway', 'ligand_receptor_pair'],
            'prob': ['prob', 'probability', 'pval', 'p_value', 'score', 'strength']
        }

        # 应用列名映射
        for standard_name, alt_names in col_mapping.items():
            for alt in alt_names:
                if alt in cellchat_comm.columns:
                    cellchat_comm.rename(columns={alt: standard_name}, inplace=True)
                    break

        # 确保有必要的列
        required_columns = ['source', 'target']
        if not all(col in cellchat_comm.columns for col in required_columns):
            missing = [col for col in required_columns if col not in cellchat_comm.columns]
            logger.warning(f"通讯文件缺少必要的列: {', '.join(missing)}")
            cellchat_comm = pd.DataFrame(columns=['source', 'target', 'interaction_name', 'prob'])
        else:
            # 确保有interaction_name和prob列
            if 'interaction_name' not in cellchat_comm.columns:
                cellchat_comm['interaction_name'] = 'Unknown'
            if 'prob' not in cellchat_comm.columns:
                cellchat_comm['prob'] = 0.0

            # 确保概率是数值
            cellchat_comm['prob'] = pd.to_numeric(cellchat_comm['prob'], errors='coerce').fillna(0)

        # 重命名列
        cellchat_comm = cellchat_comm[['source', 'target', 'interaction_name', 'prob']]
        cellchat_comm.columns = ['Sender', 'Receiver', 'Interaction', 'CellChat_Prob']
        results['communication'] = cellchat_comm
    except Exception as e:
        logger.error(f"加载通讯文件失败: {str(e)}")
        results['communication'] = pd.DataFrame()

    # 3. 加载信号通路结果
    pathways_path = os.path.join(cellchat_dir, "cellchat_pathways.csv")
    if not os.path.exists(pathways_path):
        # 尝试其他可能的文件名
        alt_names = ['pathways.csv', 'signaling_pathways.csv', 'pathway_results.csv']
        for alt in alt_names:
            alt_path = os.path.join(cellchat_dir, alt)
            if os.path.exists(alt_path):
                pathways_path = alt_path
                break
        else:
            logger.warning("未找到单独的通路文件，将从通讯文件中提取通路信息")
            pathways_path = None

    if pathways_path:
        logger.info(f"加载通路文件: {pathways_path}")
        try:
            cellchat_pathways = pd.read_csv(pathways_path)

            # 检查并重命名列
            col_mapping = {
                'pathway_name': ['pathway_name', 'pathway', 'signaling_pathway'],
                'communication_prob': ['communication_prob', 'prob', 'pval', 'score']
            }

            for standard_name, alt_names in col_mapping.items():
                for alt in alt_names:
                    if alt in cellchat_pathways.columns:
                        cellchat_pathways.rename(columns={alt: standard_name}, inplace=True)
                        break

            # 确保有必要的列
            if 'pathway_name' not in cellchat_pathways.columns:
                cellchat_pathways['pathway_name'] = 'Unknown'
            if 'communication_prob' not in cellchat_pathways.columns:
                cellchat_pathways['communication_prob'] = 0.0

            # 确保概率是数值
            cellchat_pathways['communication_prob'] = pd.to_numeric(
                cellchat_pathways['communication_prob'], errors='coerce'
            ).fillna(0)

            cellchat_pathways = cellchat_pathways[['pathway_name', 'communication_prob']]
            cellchat_pathways.columns = ['Pathway', 'Pathway_Prob']
            results['pathways'] = cellchat_pathways
        except Exception as e:
            logger.error(f"加载通路文件失败: {str(e)}")
            results['pathways'] = pd.DataFrame()
    else:
        # 尝试从通讯文件中提取通路信息
        if not results['communication'].empty and 'Interaction' in results['communication'].columns:
            logger.info("从通讯文件中提取通路信息")
            try:
                pathway_summary = results['communication'].groupby('Interaction').agg(
                    Pathway_Prob=('CellChat_Prob', 'mean')
                ).reset_index()
                pathway_summary.columns = ['Pathway', 'Pathway_Prob']
                results['pathways'] = pathway_summary
            except:
                results['pathways'] = pd.DataFrame()
        else:
            results['pathways'] = pd.DataFrame()

    return results


def load_cellphonedb_results(cpdb_dir):
    """
    加载CellPhoneDB结果文件
    """
    logger.info("加载CellPhoneDB结果...")
    # 查找significant_means文件
    sig_means_files = [
        'significant_means.txt', 'significant_means.csv', 'significant_means.tsv',
        'significant_means.xlsx', 'deconvoluted.txt', 'means.txt'
    ]

    sig_means_path = None
    for file_name in sig_means_files:
        test_path = os.path.join(cpdb_dir, file_name)
        if os.path.exists(test_path):
            sig_means_path = test_path
            break

    if not sig_means_path:
        raise FileNotFoundError(f"在目录 {cpdb_dir} 中找不到CellPhoneDB结果文件")

    logger.info(f"加载CellPhoneDB文件: {sig_means_path}")

    # 尝试不同的读取方法
    try:
        # 尝试读取文本文件
        if sig_means_path.endswith('.txt') or sig_means_path.endswith('.tsv'):
            cpdb_sig = pd.read_csv(sig_means_path, sep='\t')
        elif sig_means_path.endswith('.csv'):
            cpdb_sig = pd.read_csv(sig_means_path)
        elif sig_means_path.endswith('.xlsx'):
            cpdb_sig = pd.read_excel(sig_means_path)
        else:
            # 尝试自动检测分隔符
            with open(sig_means_path, 'r') as f:
                first_line = f.readline()

            if '\t' in first_line:
                cpdb_sig = pd.read_csv(sig_means_path, sep='\t')
            elif ',' in first_line:
                cpdb_sig = pd.read_csv(sig_means_path)
            else:
                cpdb_sig = pd.read_csv(sig_means_path, sep=None, engine='python')
    except Exception as e:
        logger.error(f"解析文件失败: {str(e)}")
        raise

    # 处理CellPhoneDB数据
    # 获取所有细胞对列名
    non_pair_cols = ['id_cp_interaction', 'interacting_pair', 'gene_a', 'gene_b', 'partner_a', 'partner_b']
    cell_pair_cols = [col for col in cpdb_sig.columns if col not in non_pair_cols]

    # 如果没有找到细胞对列，尝试其他方法
    if not cell_pair_cols:
        # 尝试识别包含"|"的列
        cell_pair_cols = [col for col in cpdb_sig.columns if '|' in col]

        # 如果还是没有，使用所有数值列
        if not cell_pair_cols:
            numeric_cols = cpdb_sig.select_dtypes(include=np.number).columns
            cell_pair_cols = [col for col in numeric_cols if col not in non_pair_cols]

    if not cell_pair_cols:
        raise ValueError("无法识别细胞对列")

    logger.info(f"识别到 {len(cell_pair_cols)} 个细胞对列")

    # 转换成长格式
    cpdb_long = pd.melt(
        cpdb_sig,
        id_vars=['gene_a', 'gene_b'] if 'gene_a' in cpdb_sig.columns else non_pair_cols,
        value_vars=cell_pair_cols,
        var_name='CellPair',
        value_name='CPDB_Score'
    )

    # 确保分数是数值
    cpdb_long['CPDB_Score'] = pd.to_numeric(cpdb_long['CPDB_Score'], errors='coerce')

    # 过滤无效值
    cpdb_long = cpdb_long.dropna(subset=['CPDB_Score'])
    cpdb_long = cpdb_long[cpdb_long['CPDB_Score'] > 0]

    # 拆分细胞对
    try:
        cpdb_long[['Sender', 'Receiver']] = cpdb_long['CellPair'].str.split(r'\|', expand=True)
    except:
        # 尝试其他分隔符
        try:
            cpdb_long[['Sender', 'Receiver']] = cpdb_long['CellPair'].str.split(r'[|;:]', expand=True)
        except:
            # 如果无法拆分，使用整个字符串作为发送者
            cpdb_long['Sender'] = cpdb_long['CellPair']
            cpdb_long['Receiver'] = 'Unknown'

    # 聚合结果
    cpdb_agg = cpdb_long.groupby(['Sender', 'Receiver']).agg(
        CPDB_Mean=('CPDB_Score', 'mean'),
        CPDB_Max=('CPDB_Score', 'max'),
        Num_Interactions=('CPDB_Score', 'count')
    ).reset_index()

    return cpdb_agg


def integrate_results(cellchat_results, cpdb_results):
    """
    整合CellChat和CellPhoneDB结果
    """
    logger.info("整合结果...")
    # 合并CellChat矩阵和CellPhoneDB结果
    if 'matrix' in cellchat_results and not cellchat_results['matrix'].empty:
        combined = pd.merge(
            cellchat_results['matrix'],
            cpdb_results,
            on=['Sender', 'Receiver'],
            how='outer'
        )
    else:
        # 如果没有CellChat矩阵，使用CellPhoneDB结果
        combined = cpdb_results.copy()
        combined['CellChat_Score'] = 0

    # 填充缺失值
    combined['CellChat_Score'] = combined['CellChat_Score'].fillna(0)
    combined['CPDB_Mean'] = combined['CPDB_Mean'].fillna(0)
    combined['CPDB_Max'] = combined['CPDB_Max'].fillna(0)
    combined['Num_Interactions'] = combined['Num_Interactions'].fillna(0)

    # 归一化分数 - 使用MinMaxScaler归一化到[0,1]范围
    scaler = MinMaxScaler()

    if combined['CellChat_Score'].nunique() > 1:
        combined['Norm_CellChat'] = scaler.fit_transform(combined[['CellChat_Score']])
    else:
        combined['Norm_CellChat'] = combined['CellChat_Score']

    if combined['CPDB_Mean'].nunique() > 1:
        combined['Norm_CPDB_Mean'] = scaler.fit_transform(combined[['CPDB_Mean']])
    else:
        combined['Norm_CPDB_Mean'] = combined['CPDB_Mean']

    if combined['CPDB_Max'].nunique() > 1:
        combined['Norm_CPDB_Max'] = scaler.fit_transform(combined[['CPDB_Max']])
    else:
        combined['Norm_CPDB_Max'] = combined['CPDB_Max']

    # 计算共识分数
    combined['Consensus_Score'] = (
                                          combined['Norm_CellChat'] +
                                          combined['Norm_CPDB_Mean'] +
                                          combined['Norm_CPDB_Max']
                                  ) / 3
    combined['Consensus_Score'] = combined['Consensus_Score'].fillna(0)

    # 添加CellChat通信概率（如果可用）
    if 'communication' in cellchat_results and not cellchat_results['communication'].empty:
        cellchat_comm_summary = cellchat_results['communication'].groupby(['Sender', 'Receiver']).agg(
            CellChat_MeanProb=('CellChat_Prob', 'mean'),
            CellChat_MaxProb=('CellChat_Prob', 'max')
        ).reset_index()

        combined = pd.merge(
            combined,
            cellchat_comm_summary,
            on=['Sender', 'Receiver'],
            how='left'
        )

        combined['CellChat_MeanProb'] = combined['CellChat_MeanProb'].fillna(0)
        combined['CellChat_MaxProb'] = combined['CellChat_MaxProb'].fillna(0)
    else:
        combined['CellChat_MeanProb'] = 0
        combined['CellChat_MaxProb'] = 0

    # 添加信号通路信息（如果可用）
    if 'pathways' in cellchat_results and not cellchat_results['pathways'].empty:
        # 获取前5个通路
        top_pathways = cellchat_results['pathways'].sort_values('Pathway_Prob', ascending=False).head(5)
        top_pathway_names = top_pathways['Pathway'].tolist()

        # 从通讯文件中提取涉及这些通路的互作
        if 'communication' in cellchat_results and not cellchat_results['communication'].empty:
            pathway_interactions = cellchat_results['communication']
            if 'Interaction' in pathway_interactions.columns:
                pathway_interactions = pathway_interactions[
                    pathway_interactions['Interaction'].isin(top_pathway_names)
                ]

                if not pathway_interactions.empty:
                    pathway_summary = pathway_interactions.groupby(['Sender', 'Receiver']).agg(
                        Top_Pathways=('Interaction', lambda x: ';'.join(set(x))),
                        Pathway_Score=('CellChat_Prob', 'max')
                    ).reset_index()

                    combined = pd.merge(
                        combined,
                        pathway_summary,
                        on=['Sender', 'Receiver'],
                        how='left'
                    )

                    combined['Pathway_Score'] = combined['Pathway_Score'].fillna(0)
                    combined['Top_Pathways'] = combined['Top_Pathways'].fillna('')
                else:
                    combined['Top_Pathways'] = ''
                    combined['Pathway_Score'] = 0
            else:
                combined['Top_Pathways'] = ''
                combined['Pathway_Score'] = 0
        else:
            combined['Top_Pathways'] = ''
            combined['Pathway_Score'] = 0
    else:
        combined['Top_Pathways'] = ''
        combined['Pathway_Score'] = 0

    return combined


def generate_full_interaction_matrix(integrated_data):
    """
    生成完整的细胞互作矩阵
    """
    logger.info("生成完整的细胞互作矩阵...")

    # 获取所有唯一的细胞类型
    all_cell_types = sorted(set(integrated_data['Sender'].unique()) | set(integrated_data['Receiver'].unique()))

    # 创建所有可能的细胞对组合
    all_pairs = pd.DataFrame(
        [(sender, receiver) for sender in all_cell_types for receiver in all_cell_types if sender != receiver],
        columns=['Sender', 'Receiver']
    )

    # 合并现有数据
    full_matrix = pd.merge(
        all_pairs,
        integrated_data,
        on=['Sender', 'Receiver'],
        how='left'
    )

    # 填充缺失值
    full_matrix['Consensus_Score'] = full_matrix['Consensus_Score'].fillna(0)
    full_matrix['CellChat_Score'] = full_matrix['CellChat_Score'].fillna(0)
    full_matrix['CPDB_Mean'] = full_matrix['CPDB_Mean'].fillna(0)
    full_matrix['CPDB_Max'] = full_matrix['CPDB_Max'].fillna(0)
    full_matrix['Num_Interactions'] = full_matrix['Num_Interactions'].fillna(0)
    full_matrix['CellChat_MeanProb'] = full_matrix['CellChat_MeanProb'].fillna(0)
    full_matrix['CellChat_MaxProb'] = full_matrix['CellChat_MaxProb'].fillna(0)
    full_matrix['Pathway_Score'] = full_matrix['Pathway_Score'].fillna(0)
    full_matrix['Top_Pathways'] = full_matrix['Top_Pathways'].fillna('')

    # 添加互作对ID
    full_matrix['Pair_ID'] = full_matrix['Sender'] + '_' + full_matrix['Receiver']

    return full_matrix


def assign_gold_standard_labels(full_matrix, threshold_method="quantile", threshold_value=0.7):
    """
    为所有细胞互作关系分配金标准标签
    threshold_method: "quantile" (分位数) 或 "absolute" (绝对阈值)
    threshold_value: 分位数阈值(0-1) 或绝对阈值分数
    """
    logger.info("为所有细胞互作关系分配金标准标签...")

    if full_matrix.empty:
        logger.warning("互作矩阵为空，无法分配标签")
        return full_matrix

    # 创建副本以避免修改原始数据
    labeled_matrix = full_matrix.copy()

    if threshold_method == "quantile":
        # 使用分位数阈值
        threshold = labeled_matrix['Consensus_Score'].quantile(threshold_value)
        logger.info(f"使用分位数阈值: {threshold_value} -> 共识分数阈值: {threshold:.4f}")
    elif threshold_method == "absolute":
        # 使用绝对阈值
        threshold = threshold_value
        logger.info(f"使用绝对阈值: {threshold_value}")
    else:
        raise ValueError("threshold_method 必须是 'quantile' 或 'absolute'")

    # 分配金标准标签
    # 1: 高置信度正样本 (共识分数 >= 阈值)
    # 0: 负样本 (共识分数 < 阈值)
    labeled_matrix['Gold_Standard_Label'] = (labeled_matrix['Consensus_Score'] >= threshold).astype(int)

    # 计算置信度级别
    if not labeled_matrix.empty:
        # 创建置信度分级
        if threshold_method == "quantile":
            # 对于分位数方法，使用动态分箱
            high_threshold = threshold
            medium_threshold = labeled_matrix['Consensus_Score'].quantile(threshold_value * 0.7)
            low_threshold = labeled_matrix['Consensus_Score'].quantile(threshold_value * 0.3)

            bins = [-np.inf, low_threshold, medium_threshold, high_threshold, np.inf]
            labels = ['Very Low', 'Low', 'Medium', 'High']
        else:
            # 对于绝对阈值方法，使用固定分箱
            high_threshold = threshold
            medium_threshold = threshold * 0.7
            low_threshold = threshold * 0.3

            bins = [-np.inf, low_threshold, medium_threshold, high_threshold, np.inf]
            labels = ['Very Low', 'Low', 'Medium', 'High']

        # 先创建基础的置信度级别
        labeled_matrix['Confidence_Level'] = pd.cut(
            labeled_matrix['Consensus_Score'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        # 将分类变量转换为字符串，以便添加新类别
        labeled_matrix['Confidence_Level'] = labeled_matrix['Confidence_Level'].astype(str)

        # 为金标准标签为1的样本添加"Gold Standard"标签
        labeled_matrix.loc[labeled_matrix['Gold_Standard_Label'] == 1, 'Confidence_Level'] = 'Gold Standard'

        # 处理可能的NaN值（由于分箱边界问题）
        labeled_matrix['Confidence_Level'] = labeled_matrix['Confidence_Level'].fillna('Very Low')
    else:
        labeled_matrix['Confidence_Level'] = 'Unknown'

    # 统计信息
    num_total = len(labeled_matrix)
    num_positive = labeled_matrix['Gold_Standard_Label'].sum()
    num_negative = num_total - num_positive

    logger.info(f"总互作关系数: {num_total}")
    logger.info(f"金标准正样本数: {num_positive} ({num_positive / num_total * 100:.1f}%)")
    logger.info(f"负样本数: {num_negative} ({num_negative / num_total * 100:.1f}%)")

    return labeled_matrix


def create_machine_learning_dataset(labeled_matrix, include_features=True):
    """
    创建适用于机器学习的数据集
    """
    logger.info("创建机器学习数据集...")

    if labeled_matrix.empty:
        logger.warning("标记矩阵为空，无法创建数据集")
        return pd.DataFrame()

    # 创建机器学习特征集
    ml_dataset = labeled_matrix[['Sender', 'Receiver', 'Pair_ID', 'Gold_Standard_Label', 'Confidence_Level']].copy()

    if include_features:
        # 包含所有可用的特征
        feature_columns = [
            'Consensus_Score', 'CellChat_Score', 'CPDB_Mean', 'CPDB_Max',
            'Num_Interactions', 'CellChat_MeanProb', 'CellChat_MaxProb', 'Pathway_Score'
        ]

        # 只添加存在的列
        available_features = [col for col in feature_columns if col in labeled_matrix.columns]
        for feature in available_features:
            ml_dataset[feature] = labeled_matrix[feature]

        logger.info(f"包含的特征: {', '.join(available_features)}")

    # 添加细胞类型编码（用于机器学习模型）
    all_cell_types = sorted(set(labeled_matrix['Sender'].unique()) | set(labeled_matrix['Receiver'].unique()))
    cell_type_map = {cell_type: idx for idx, cell_type in enumerate(all_cell_types)}

    ml_dataset['Sender_Encoded'] = ml_dataset['Sender'].map(cell_type_map)
    ml_dataset['Receiver_Encoded'] = ml_dataset['Receiver'].map(cell_type_map)

    logger.info(f"总细胞类型数: {len(all_cell_types)}")
    logger.info(f"机器学习数据集形状: {ml_dataset.shape}")

    return ml_dataset


def visualize_all_interactions(labeled_matrix, output_dir):
    """
    可视化所有细胞互作关系
    """
    logger.info("可视化所有细胞互作关系...")
    os.makedirs(output_dir, exist_ok=True)

    if labeled_matrix.empty:
        logger.warning("标记矩阵为空，跳过可视化")
        return

    try:
        # 1. 完整互作矩阵热图
        heatmap_data = labeled_matrix.pivot_table(
            index='Sender',
            columns='Receiver',
            values='Consensus_Score',
            fill_value=0
        )

        # 对行列进行排序
        heatmap_data = heatmap_data.reindex(
            index=sorted(heatmap_data.index),
            columns=sorted(heatmap_data.columns)
        )

        plt.figure(figsize=(16, 14))
        sns.heatmap(heatmap_data, cmap='Reds', annot=True, fmt=".3f", linewidths=.5,
                    cbar_kws={'label': 'Consensus Score'})
        plt.title('Complete Cell-Cell Interaction Matrix (All Pairs)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'complete_interaction_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 金标准标签热图
        label_heatmap = labeled_matrix.pivot_table(
            index='Sender',
            columns='Receiver',
            values='Gold_Standard_Label',
            fill_value=0
        )

        label_heatmap = label_heatmap.reindex(
            index=sorted(label_heatmap.index),
            columns=sorted(label_heatmap.columns)
        )

        plt.figure(figsize=(16, 14))
        sns.heatmap(label_heatmap, cmap=['lightgray', 'red'], annot=True, fmt="d",
                    linewidths=.5, cbar_kws={'label': 'Gold Standard Label (0/1)'})
        plt.title('Gold Standard Labels for All Cell-Cell Interactions')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gold_standard_labels.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 共识分数分布图
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(labeled_matrix['Consensus_Score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Consensus Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Consensus Scores')
        plt.axvline(labeled_matrix['Consensus_Score'].median(), color='red', linestyle='--',
                    label=f'Median: {labeled_matrix["Consensus_Score"].median():.3f}')
        plt.legend()

        plt.subplot(2, 2, 2)
        gold_standard = labeled_matrix[labeled_matrix['Gold_Standard_Label'] == 1]
        non_gold = labeled_matrix[labeled_matrix['Gold_Standard_Label'] == 0]

        plt.hist(gold_standard['Consensus_Score'], bins=30, alpha=0.7, label='Gold Standard', color='red')
        plt.hist(non_gold['Consensus_Score'], bins=30, alpha=0.7, label='Non-Gold', color='gray')
        plt.xlabel('Consensus Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution by Label')
        plt.legend()

        plt.subplot(2, 2, 3)
        label_counts = labeled_matrix['Gold_Standard_Label'].value_counts()
        plt.pie(label_counts.values, labels=['Negative (0)', 'Positive (1)'],
                autopct='%1.1f%%', colors=['lightgray', 'lightcoral'])
        plt.title('Gold Standard Label Distribution')

        plt.subplot(2, 2, 4)
        confidence_counts = labeled_matrix['Confidence_Level'].value_counts()
        plt.bar(confidence_counts.index.astype(str), confidence_counts.values,
                color=['lightgray', 'lightblue', 'skyblue', 'deepskyblue', 'red'])
        plt.xlabel('Confidence Level')
        plt.ylabel('Count')
        plt.title('Confidence Level Distribution')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 网络图
        gold_standard_pairs = labeled_matrix[labeled_matrix['Gold_Standard_Label'] == 1]

        if not gold_standard_pairs.empty:
            plt.figure(figsize=(14, 12))
            G = nx.from_pandas_edgelist(
                gold_standard_pairs,
                'Sender',
                'Receiver',
                edge_attr='Consensus_Score',
                create_using=nx.DiGraph()
            )

            # 计算节点大小（基于度中心性）
            degrees = dict(G.degree)
            node_sizes = [v * 500 for v in degrees.values()]

            # 绘制网络图
            pos = nx.spring_layout(G, k=1, iterations=50)
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color='lightblue', alpha=0.9)
            nx.draw_networkx_edges(G, pos, edge_color='red',
                                   width=2.0, alpha=0.7, arrows=True, arrowsize=20)
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

            plt.title(f'Gold Standard Cell Interaction Network ({len(gold_standard_pairs)} pairs)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gold_standard_network.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        logger.error(f"可视化失败: {str(e)}")


def create_complete_gold_standard_dataset(cellchat_dir, cpdb_dir, output_dir,
                                          threshold_method="quantile", threshold_value=0.7):
    """
    创建完整的金标准数据集（为所有细胞互作关系建立标签）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        cellchat_results = load_cellchat_results(cellchat_dir)
    except Exception as e:
        logger.error(f"加载CellChat结果时出错: {str(e)}")
        return None

    try:
        cpdb_results = load_cellphonedb_results(cpdb_dir)
    except Exception as e:
        logger.error(f"加载CellPhoneDB结果时出错: {str(e)}")
        return None

    # 整合结果
    integrated_data = integrate_results(cellchat_results, cpdb_results)

    # 生成完整的互作矩阵
    full_matrix = generate_full_interaction_matrix(integrated_data)

    # 为所有互作关系分配金标准标签
    labeled_matrix = assign_gold_standard_labels(
        full_matrix,
        threshold_method=threshold_method,
        threshold_value=threshold_value
    )

    # 创建机器学习数据集
    ml_dataset = create_machine_learning_dataset(labeled_matrix, include_features=True)

    # 保存所有结果
    # 1. 完整标记矩阵
    full_output_path = os.path.join(output_dir, 'complete_labeled_interactions.csv')
    labeled_matrix.to_csv(full_output_path, index=False)
    logger.info(f"完整标记矩阵已保存: {full_output_path}")

    # 2. 机器学习数据集
    ml_output_path = os.path.join(output_dir, 'machine_learning_dataset.csv')
    ml_dataset.to_csv(ml_output_path, index=False)
    logger.info(f"机器学习数据集已保存: {ml_output_path}")

    # 3. 只包含金标准正样本的数据集（用于传统分析）
    gold_standard_only = labeled_matrix[labeled_matrix['Gold_Standard_Label'] == 1]
    gold_output_path = os.path.join(output_dir, 'gold_standard_interactions.csv')
    gold_standard_only.to_csv(gold_output_path, index=False)
    logger.info(f"金标准互作已保存: {gold_output_path}")

    # 可视化结果
    visualize_all_interactions(labeled_matrix, output_dir)

    # 保存数据集统计信息
    stats = {
        'total_interactions': len(labeled_matrix),
        'gold_standard_positive': len(gold_standard_only),
        'gold_standard_negative': len(labeled_matrix) - len(gold_standard_only),
        'positive_percentage': len(gold_standard_only) / len(labeled_matrix) * 100,
        'threshold_method': threshold_method,
        'threshold_value': threshold_value,
        'num_cell_types': len(set(labeled_matrix['Sender'].unique()) | set(labeled_matrix['Receiver'].unique()))
    }

    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(output_dir, 'dataset_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"数据集统计信息已保存: {stats_path}")

    logger.info(f"\n✅ 金标准数据集创建成功！")
    logger.info(f"总互作关系数: {len(labeled_matrix)}")
    logger.info(f"金标准正样本数: {len(gold_standard_only)}")
    logger.info(f"正样本比例: {stats['positive_percentage']:.1f}%")
    logger.info(f"细胞类型数: {stats['num_cell_types']}")
    logger.info(f"阈值方法: {threshold_method}, 阈值: {threshold_value}")

    return {
        'labeled_matrix': labeled_matrix,
        'ml_dataset': ml_dataset,
        'gold_standard_only': gold_standard_only,
        'statistics': stats
    }


if __name__ == "__main__":
    # 示例使用 - 修改为您的实际路径
    cellchat_dir = "./CellChat_Results/"
    cpdb_dir = "./CellphoneDB_results/"
    output_dir = "./complete_gold_standard/"
    os.makedirs(output_dir, exist_ok=True)

    # 创建金标准数据集
    results = create_complete_gold_standard_dataset(
        cellchat_dir=cellchat_dir,
        cpdb_dir=cpdb_dir,
        output_dir=output_dir,
        threshold_method="quantile",
        threshold_value=0.7
    )

    if results is not None:
        logger.info("数据集创建完成！")
        logger.info("输出文件包括:")
        logger.info("1. complete_labeled_interactions.csv - 完整标记矩阵")
        logger.info("2. machine_learning_dataset.csv - 机器学习数据集")
        logger.info("3. gold_standard_interactions.csv - 仅金标准正样本")
        logger.info("4. dataset_statistics.csv - 数据集统计信息")
        logger.info("5. 多种可视化图表")
