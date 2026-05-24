# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys

# Windows 终端控制台打印 utf-8 可能会报错，重新配置 stdout
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def load_and_clean_chembl_data(file_path, target_name):
    """
    读取并清洗从 ChEMBL 下载的原始 CSV 数据
    带有强容错机制，跳过格式损坏的行
    """
    print(f"正在处理 {target_name} 的数据...")

    # 1. 强力读取机制
    encodings = ['utf-8', 'latin1', 'cp1252', 'gbk']
    df = None
    for enc in encodings:
        try:
            df_temp = pd.read_csv(file_path, sep=';', on_bad_lines='skip', low_memory=False, encoding=enc)
            if 'Smiles' in df_temp.columns or 'Canonical Smiles' in df_temp.columns:
                df = df_temp
                break
        except Exception:
            pass
            
        try:
            df_temp = pd.read_csv(file_path, sep=',', on_bad_lines='skip', low_memory=False, encoding=enc)
            if 'Smiles' in df_temp.columns or 'Canonical Smiles' in df_temp.columns:
                df = df_temp
                break
        except Exception:
            pass
                
    if df is None or df.empty:
        print(f"❌ 读取 {target_name} 数据文件失败，请检查文件格式。")
        return pd.DataFrame()

    # 2. 统一列名
    if 'Canonical Smiles' in df.columns:
        df = df.rename(columns={'Canonical Smiles': 'Smiles'})

    # 3. 核心过滤规则
    if 'Smiles' not in df.columns:
        print(f"⚠️ {target_name} 数据缺少 Smiles 结构列。")
        return pd.DataFrame()

    df = df.dropna(subset=['Smiles'])

    if 'Standard Type' in df.columns:
        df = df[df['Standard Type'] == 'IC50']

    if 'Standard Units' in df.columns:
        df = df[df['Standard Units'] == 'nM']

    # 4. 提取数值并去重
    if 'Standard Value' in df.columns:
        # 强制转换为数值型，无法转换的（如 '>10000'）变为 NaN 并删除
        df['Standard Value'] = pd.to_numeric(df['Standard Value'], errors='coerce')
        df = df.dropna(subset=['Standard Value'])

        # 只保留所需列
        df = df[['Smiles', 'Standard Value']]

        # 药学数据去重：同一分子多次测试取平均值
        df = df.groupby('Smiles', as_index=False).mean()

    print(f"✅ {target_name} 清洗完成，剩余 {len(df)} 个独特分子。")
    return df

def main():
    # 原始文件路径
    file_hdac6 = 'data/HDAC6_raw.csv'

    if not os.path.exists(file_hdac6):
        print("❌ 错误：找不到原始数据文件，请确保文件名正确并已上传。")
        return

    # 执行清洗 (仅针对 HDAC6)
    df_hdac6 = load_and_clean_chembl_data(file_hdac6, "HDAC6")

    if df_hdac6.empty:
        print("❌ 错误：清洗后数据为空。")
        return

    # 重命名活性列以便区分
    df_hdac6 = df_hdac6.rename(columns={'Standard Value': 'IC50_HDAC6'})

    # ==========================================
    # 构建机器学习二分类标签 (单纯的活性预测)
    # ==========================================
    # 正样本（Label=1）定义标准：
    # 对 HDAC6 的 IC50 <= 1000 nM，无论其对其他靶点是否有作用。
    # 
    # 负样本（Label=0）定义标准：
    # 仅包含对 HDAC6 活性较差的分子 (IC50 > 1000 nM)。
    # ==========================================
    ACTIVITY_THRESHOLD = 1000

    condition_activity = df_hdac6['IC50_HDAC6'] <= ACTIVITY_THRESHOLD
    
    df_hdac6['Label'] = condition_activity.astype(int)

    print(f"🎉 活性数据集构建成功！共获得 {len(df_hdac6)} 个符合要求的分子。")
    print(f" 🔹 正样本 (HDAC6高活性抑制剂, Label=1): {df_hdac6['Label'].sum()} 个")
    print(f" 🔹 负样本 (HDAC6低活性分子, Label=0): {len(df_hdac6) - df_hdac6['Label'].sum()} 个")

    # 保存最终数据
    output_file = 'data/HDAC6_Activity_ML_Ready_Data.csv'
    df_hdac6.to_csv(output_file, index=False)
    print(f"\n💾 活性数据已保存至: {output_file}")
    print("接下来的任务是：使用此数据集提取分子结构特征并训练活性分类模型。")

if __name__ == "__main__":
    main()
