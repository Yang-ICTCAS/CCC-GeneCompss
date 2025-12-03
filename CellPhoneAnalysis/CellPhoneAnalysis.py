from cellphonedb.src.core.methods import cpdb_degs_analysis_method
import multiprocessing
import os
import pandas as pd
import logging
import scanpy as sc
import argparse
import numpy as np
from scipy.sparse import issparse
import tempfile
import shutil

# 配置详细的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_files_from_h5ad(h5ad_path, groupby='cell_type', output_dir=None):
    """从h5ad文件提取CellPhoneDB所需的counts和meta文件"""
    try:
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.dirname(h5ad_path)

        # 创建cellphonedb_input子目录
        cpdb_input_dir = os.path.join(output_dir, "cellphonedb_input")
        os.makedirs(cpdb_input_dir, exist_ok=True)
        logger.info(f"创建CellPhoneDB输入目录: {cpdb_input_dir}")

        # 读取h5ad文件
        logger.info(f"读取h5ad文件: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)

        # 检查分组列是否存在
        if groupby not in adata.obs.columns:
            raise ValueError(f"分组列 '{groupby}' 不存在于adata.obs中")

        logger.info(f"数据集信息: {adata.shape[0]}个细胞, {adata.shape[1]}个基因")

        # 1. 导出计数矩阵 (counts.txt)
        counts_path = os.path.join(cpdb_input_dir, "counts.txt")
        logger.info(f"导出计数矩阵到: {counts_path}")

        # 处理稀疏矩阵
        if issparse(adata.X):
            counts_df = pd.DataFrame(adata.X.toarray().T, index=adata.var_names, columns=adata.obs_names)
        else:
            counts_df = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)

        counts_df.to_csv(counts_path, sep='\t')
        logger.info(f"计数矩阵导出完成, 包含 {counts_df.shape[0]} 个基因和 {counts_df.shape[1]} 个细胞")

        # 2. 导出元数据 (meta.txt)
        meta_path = os.path.join(cpdb_input_dir, "meta.txt")
        logger.info(f"导出元数据到: {meta_path}")

        # 创建元数据DataFrame
        meta_df = adata.obs[[groupby]].copy()
        meta_df.index.name = 'Cell'
        meta_df.columns = ['cell_type']
        meta_df.to_csv(meta_path, sep='\t')

        # 统计细胞类型分布
        cell_type_counts = meta_df['cell_type'].value_counts()
        logger.info("细胞类型分布:")
        for cell_type, count in cell_type_counts.items():
            logger.info(f"  {cell_type}: {count} 个细胞")

        return counts_path, meta_path

    except Exception as e:
        logger.error(f"从h5ad文件提取文件时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def prepare_degs_file_for_cpdb(degs_file_path):
    """准备DEGs文件以供CellPhoneDB使用"""
    try:
        logger.info(f"准备DEGs文件: {degs_file_path}")

        # 读取DEGs文件
        degs_df = pd.read_csv(degs_file_path, sep='\t')

        # 检查必要的列是否存在
        required_columns = ['cluster', 'gene']
        missing_columns = [col for col in required_columns if col not in degs_df.columns]

        if missing_columns:
            logger.error(f"DEGs文件缺少必要的列: {', '.join(missing_columns)}")
            return None

        # 创建临时目录和文件
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "cpdb_degs.tsv")

        # 创建只有两列的新文件
        cpdb_degs_df = degs_df[['cluster', 'gene']].copy()
        cpdb_degs_df.to_csv(temp_file, sep='\t', index=False)

        logger.info(f"为CellPhoneDB创建临时DEGs文件: {temp_file}")
        logger.info(f"包含 {len(cpdb_degs_df)} 个差异表达基因")

        return temp_file

    except Exception as e:
        logger.error(f"准备DEGs文件时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_cellphonedb_analysis(cpdb_zip, h5ad_path, degs_file, microenvs_file,
                             groupby='cell_type', output_dir=None):
    """使用h5ad文件和已有的DEGs、微环境文件运行CellPhoneDB分析"""
    try:
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.dirname(h5ad_path)

        # 1. 从h5ad文件提取counts和meta文件
        counts_path, meta_path = extract_files_from_h5ad(
            h5ad_path, groupby, output_dir
        )

        if counts_path is None or meta_path is None:
            logger.error("提取输入文件失败，分析终止")
            return False

        # 2. 准备DEGs文件
        cpdb_degs_path = prepare_degs_file_for_cpdb(degs_file)
        if cpdb_degs_path is None:
            logger.error("准备DEGs文件失败，分析终止")
            return False

        # 3. 详细检查所有文件存在性
        logger.info("检查所有必需文件是否存在...")
        required_files = [cpdb_zip, counts_path, meta_path, microenvs_file]
        all_files_exist = True

        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"错误: 文件不存在 - {file}")
                all_files_exist = False
            else:
                logger.info(f"找到文件: {file} - 大小: {os.path.getsize(file) / 1024:.1f} KB")

        if not all_files_exist:
            logger.error("缺少必要文件，分析终止")
            return False

        # 4. 验证微环境文件格式
        try:
            logger.info("验证微环境文件格式...")
            microenvs_df = pd.read_csv(microenvs_file, sep='\t', header=None)
            logger.info(f"微环境文件包含 {len(microenvs_df)} 行")

            if len(microenvs_df.columns) < 2:
                logger.error("微环境文件格式错误：列数不足")
                return False
            else:
                # 检查是否有重复的细胞类型
                unique_combinations = microenvs_df[[0, 1]].drop_duplicates()
                if len(unique_combinations) < len(microenvs_df):
                    logger.warning(f"微环境文件包含 {len(microenvs_df) - len(unique_combinations)} 个重复条目")

                logger.info(f"微环境文件格式验证通过: {len(microenvs_df.columns)} 列, {len(microenvs_df)} 行")
                logger.info("微环境文件前5行预览:")
                for i in range(min(5, len(microenvs_df))):
                    logger.info(f"行 {i + 1}: {microenvs_df.iloc[i].values}")
        except Exception as e:
            logger.error(f"验证微环境文件时出错: {str(e)}")
            return False

        # 5. 验证计数文件
        try:
            logger.info("验证计数文件...")
            counts_df = pd.read_csv(counts_path, sep='\t', index_col=0)
            logger.info(f"计数文件包含 {counts_df.shape[0]} 个基因, {counts_df.shape[1]} 个细胞")

            # 检查是否有空值
            if counts_df.isnull().any().any():
                logger.warning(f"计数文件中有 {counts_df.isnull().sum().sum()} 个空值")

            # 检查是否有零表达基因
            zero_expression_genes = counts_df.sum(axis=1) == 0
            if zero_expression_genes.any():
                logger.warning(f"有 {zero_expression_genes.sum()} 个基因在所有细胞中表达量为零")

            # 打印前5个基因名
            logger.info(f"前5个基因: {counts_df.index[:5].tolist()}")

        except Exception as e:
            logger.error(f"验证计数文件时出错: {str(e)}")
            return False

        # 6. 创建CellPhoneDB输出目录
        cpdb_output_dir = os.path.join(output_dir, "cellphonedb_results_20251016")
        os.makedirs(cpdb_output_dir, exist_ok=True)
        logger.info(f"创建CellPhoneDB输出目录: {cpdb_output_dir}")

        # 7. 尝试不同的计数数据标识符
        possible_count_types = ['gene_name', 'hgnc_symbol', 'ensembl']
        success = False

        for count_type in possible_count_types:
            try:
                logger.info(f"尝试使用计数数据类型: {count_type}")
                logger.info("开始CellPhoneDB分析...")

                # 运行CellPhoneDB分析
                cpdb_degs_analysis_method.call(
                    cpdb_file_path=cpdb_zip,
                    meta_file_path=meta_path,
                    counts_file_path=counts_path,
                    degs_file_path=cpdb_degs_path,
                    counts_data=count_type,
                    score_interactions=True,
                    threshold=0.01,  # 使用更宽松的阈值
                    output_path=cpdb_output_dir
                )

                logger.info("CellPhoneDB分析成功完成!")
                success = True
                break

            except Exception as e:
                logger.warning(f"使用 '{count_type}' 时分析失败: {str(e)}")
                logger.warning("尝试下一种基因标识符类型...")

        # 8. 清理临时文件
        try:
            temp_dir = os.path.dirname(cpdb_degs_path)
            shutil.rmtree(temp_dir)
            logger.info(f"已清理临时目录: {temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {str(e)}")

        # 如果所有尝试都失败
        if not success:
            logger.error("所有基因标识符类型尝试均失败")
            return False

        return True

    except Exception as e:
        logger.error(f"运行CellPhoneDB分析时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='从h5ad文件进行细胞互作分析')
    parser.add_argument('--h5ad',
                        default=r"G:\DATA\SingleCell\TabulaSapiens\tabula_sapiens_liver\c264e09f-7c3b-4294-b0f4-82a790bd0014.h5ad",
                        type=str, help='h5ad文件路径')
    parser.add_argument('--cpdb', default="./v5.0.0/cellphonedb.zip", type=str, help='CellPhoneDB数据库zip文件路径')
    parser.add_argument('--degs', default="./liver_DEGs_wilcoxon.tsv", type=str, help='差异表达基因文件路径')
    parser.add_argument('--microenv', default="./liver_microenvironment.tsv", type=str, help='微环境文件路径')
    parser.add_argument('--groupby', default='cell_type', type=str, help='分组列名 (默认: cell_type)')
    parser.add_argument('--outdir', default="./", type=str, help='输出目录 (默认: 当前目录)')
    args = parser.parse_args()

    logger.info("开始细胞互作分析流程")
    success = run_cellphonedb_analysis(
        cpdb_zip=args.cpdb,
        h5ad_path=args.h5ad,
        degs_file=args.degs,
        microenvs_file=args.microenv,
        groupby=args.groupby,
        output_dir=args.outdir
    )

    if success:
        logger.info("分析成功完成!")
        # 打印结果文件位置
        results_dir = os.path.join(args.outdir, "cellphonedb_results_20251016")
        logger.info(f"CellPhoneDB结果文件保存在: {results_dir}")
        logger.info(f"重要文件: deconvoluted.txt, means.txt, pvalues.txt, significant_means.txt")
    else:
        logger.error("分析失败。请检查错误信息。")