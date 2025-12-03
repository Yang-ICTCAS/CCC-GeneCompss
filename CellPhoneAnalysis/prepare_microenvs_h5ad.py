import pandas as pd
import scanpy as sc
import argparse
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_microenv_file(h5ad_path, groupby='cell_type', microenvironment_name='global_microenvironment',
                         output_dir=None):
    """
    从h5ad文件创建微环境文件

    参数:
    h5ad_path: h5ad文件路径
    groupby: 指定分组列名 (默认为'cell_type')
    microenvironment_name: 微环境名称 (默认为'global_microenvironment')
    output_dir: 输出目录 (默认为h5ad文件所在目录)
    """
    try:
        logger.info(f"读取h5ad文件: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)

        # 检查分组列是否存在
        if groupby not in adata.obs.columns:
            raise ValueError(f"分组列 '{groupby}' 不存在于adata.obs中")

        logger.info(f"数据集信息: {adata.shape[0]}个细胞, {adata.shape[1]}个基因")
        logger.info(f"分组列: '{groupby}'")

        # 获取所有唯一的细胞类型
        cell_types = adata.obs[groupby].unique()
        logger.info(f"识别到{len(cell_types)}种细胞类型")

        # 创建微环境数据
        microenvs_list = []
        for cell_type in cell_types:
            microenvs_list.append({
                'microenvironment': microenvironment_name,
                'cell_type': cell_type
            })

        # 创建DataFrame
        microenvs_df = pd.DataFrame(microenvs_list)

        # 设置输出路径
        if output_dir is None:
            output_dir = os.path.dirname(h5ad_path)
        os.makedirs(output_dir, exist_ok=True)

        # 生成输出文件名
        sample_name = os.path.splitext(os.path.basename(h5ad_path))[0]
        output_path = os.path.join(output_dir, f"{sample_name}_microenvironment.tsv")

        # 保存为TSV文件（无表头）
        microenvs_df.to_csv(output_path, sep='\t', index=False, header=False)

        logger.info(f"微环境文件已创建: {output_path}")
        logger.info(f"文件内容预览:\n{microenvs_df.head()}")

        return microenvs_df

    except Exception as e:
        logger.error(f"创建微环境文件时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    import argparse

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='从h5ad文件创建微环境文件')
    parser.add_argument('--h5ad', type=str, default=r"G:\DATA\SingleCell\TabulaSapiens\tabula_sapiens_liver\c264e09f-7c3b-4294-b0f4-82a790bd0014.h5ad", help='输入h5ad文件路径')
    parser.add_argument('--groupby', type=str, default='cell_type', help='分组列名 (默认: cell_type)')
    parser.add_argument('--env_name', type=str, default='global_microenvironment',
                        help='微环境名称 (默认: global_microenvironment)')
    parser.add_argument('--outdir', default="./", type=str, help='输出目录 (默认: 输入文件所在目录)')
    args = parser.parse_args()

    logger.info("开始创建微环境文件")
    try:
        microenv_df = create_microenv_file(
            h5ad_path=args.h5ad,
            groupby=args.groupby,
            microenvironment_name=args.env_name,
            output_dir=args.outdir
        )
        logger.info(f"成功创建微环境文件，包含 {len(microenv_df)} 个细胞类型")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")