import os
import logging
import argparse
import pandas as pd
import scanpy as sc
import scipy
from cellphonedb.src.core.methods import cpdb_degs_analysis_method
import multiprocessing

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def export_cellphonedb_input_files(adata, output_dir, groupby='cell_type'):
    """从h5ad文件导出CellPhoneDB所需的计数矩阵和元数据文件"""
    try:
        # 创建输出目录
        cpdb_input_dir = os.path.join(output_dir, "cellphonedb_input")
        os.makedirs(cpdb_input_dir, exist_ok=True)

        # 1. 导出计数矩阵
        counts_path = os.path.join(cpdb_input_dir, "counts.txt")

        # 处理稀疏矩阵
        if scipy.sparse.issparse(adata.X):
            counts_df = pd.DataFrame(adata.X.toarray().T, index=adata.var_names, columns=adata.obs_names)
        else:
            counts_df = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)

        counts_df.to_csv(counts_path, sep='\t')
        logger.info(f"计数矩阵已导出至: {counts_path}")

        # 2. 导出元数据
        meta_path = os.path.join(cpdb_input_dir, "meta.txt")
        meta_df = adata.obs[[groupby]].copy()
        meta_df.index.name = 'Cell'
        meta_df.columns = ['cell_type']
        meta_df.to_csv(meta_path, sep='\t')
        logger.info(f"元数据已导出至: {meta_path}")

        return counts_path, meta_path

    except Exception as e:
        logger.error(f"导出CellPhoneDB输入文件时出错: {str(e)}")
        raise


def run_cellphonedb_analysis(h5ad_path, cpdb_zip, degs_path, microenvs_path,
                             groupby='cell_type', output_dir=None):
    """使用已有的DEGs和微环境文件运行CellPhoneDB分析"""
    try:
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.dirname(h5ad_path)

        # 1. 读取h5ad文件
        logger.info(f"读取h5ad文件: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)

        # 检查分组列是否存在
        if groupby not in adata.obs.columns:
            raise ValueError(f"分组列 '{groupby}' 不存在于adata.obs中")

        logger.info(f"数据集信息: {adata.shape[0]}个细胞, {adata.shape[1]}个基因")

        # 2. 导出CellPhoneDB所需的计数矩阵和元数据
        logger.info("导出CellPhoneDB输入文件...")
        counts_path, meta_path = export_cellphonedb_input_files(
            adata, output_dir, groupby
        )

        # 3. 检查所有必需文件是否存在
        logger.info("检查所有必需文件是否存在...")
        required_files = [cpdb_zip, counts_path, meta_path, degs_path, microenvs_path]
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"错误: 文件不存在 - {file}")
                return False

        # 4. 验证微环境文件格式
        try:
            logger.info("验证微环境文件格式...")
            microenvs_df = pd.read_csv(microenvs_path, sep='\t', header=None)
            if len(microenvs_df.columns) < 2:
                logger.error("微环境文件格式错误：列数不足")
                return False
            logger.info("微环境文件格式验证通过")
        except Exception as e:
            logger.error(f"验证微环境文件时出错: {str(e)}")
            return False

        # 5. 设置CellPhoneDB结果目录
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # 6. 运行CellPhoneDB分析
        logger.info("开始CellPhoneDB分析...")
        cpdb_degs_analysis_method.call(
            cpdb_file_path=cpdb_zip,
            meta_file_path=meta_path,
            counts_file_path=counts_path,
            degs_file_path=degs_path,
            counts_data='gene_name',
            score_interactions=True,
            threshold=0.1,
            output_path=results_dir
        )

        logger.info(f"分析完成! 结果保存在: {results_dir}")
        return True

    except Exception as e:
        logger.error(f"运行CellPhoneDB分析时出错: {str(e)}")
        return False


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用已有的DEGs和微环境文件进行CellPhoneDB分析')
    parser.add_argument('--h5ad', default=r"G:\DATA\SingleCell\TabulaSapiens\tabula_sapiens_liver\c264e09f-7c3b-4294-b0f4-82a790bd0014.h5ad", type=str, help='输入h5ad文件路径')
    parser.add_argument('--cpdb', default="./v5.0.0/cellphonedb.zip", type=str, help='CellPhoneDB数据库zip文件路径')
    parser.add_argument('--degs', default=r"./liver_DEGs_wilcoxon.tsv", type=str, help='差异表达基因文件路径')
    parser.add_argument('--microenv', default="liver_microenvironment.tsv", type=str, help='微环境文件路径')
    parser.add_argument('--groupby', default='cell_type', type=str, help='分组列名 (默认: cell_type)')
    parser.add_argument('--outdir', default="./", type=str, help='输出目录 (默认: h5ad文件所在目录)')
    args = parser.parse_args()

    logger.info("开始细胞互作分析流程")
    try:
        success = run_cellphonedb_analysis(
            h5ad_path=args.h5ad,
            cpdb_zip=args.cpdb,
            degs_path=args.degs,
            microenvs_path=args.microenv,
            groupby=args.groupby,
            output_dir=args.outdir
        )

        if success:
            logger.info("细胞互作分析成功完成!")
        else:
            logger.error("细胞互作分析失败")
    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())