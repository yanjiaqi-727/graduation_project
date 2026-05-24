import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
except ImportError:
    print("❌ 错误：请确保已安装 rdkit, matplotlib, scikit-learn。可以运行 `pip install rdkit matplotlib scikit-learn`")
    sys.exit(1)

# Windows 终端控制台打印 utf-8 可能会报错，重新配置 stdout
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 配置 matplotlib 支持中文显示和高清输出
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def smiles_to_ecfp4(smiles, radius=2, n_bits=1024):
    """
    将 SMILES 转换为 ECFP4 分子指纹。
    这里使用 1024-bit 匹配最新 XGBoost 模型的维度。
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        # 提取摩根指纹（ECFP）
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((0,))
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr, mol
    except Exception as e:
        # 容错机制：跳过 RDKit 无法解析的畸形结构
        return None, None

def main():
    print("🔬 正在拉起“老药新用”虚拟筛选 (Virtual Screening) 流程...")
    
    model_path = "ML/models/Best_HDAC6_Activity_Model.pkl"
    output_dir = "ML/virtual_screening_activity_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载已经训练好的模型
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 {model_path}，请先运行你的 model_training.py。")
        return
        
    print(f"📥 正在加载最优预测模型：{model_path}")
    best_model = joblib.load(model_path)
    
    # 获取模型训练时的特征维度，确保匹配
    expected_features = 1024
    if hasattr(best_model, "n_features_in_"):
        expected_features = best_model.n_features_in_
    print(f"   [设置特征提取维度: {expected_features} Bits]")
    
    # 2. 获取筛选库 (FDA Approved Drugs)
    print("\n🌐 正在读取本地 FDA 批准药物库...")
    file_path = "data/FDA_Approved_structures.csv"
    
    try:
        fda_drugs = pd.read_csv(file_path)
        print("✅ 成功读取本地药物库！")
        
        if 'SMILES' in fda_drugs.columns and 'Name' in fda_drugs.columns:
            # 去掉无效行和重复药物
            fda_drugs = fda_drugs.dropna(subset=['SMILES']).copy()
            fda_drugs = fda_drugs.drop_duplicates(subset=['SMILES'])
            
            # 使用 Name 和 SMILES 列
            fda_drugs = fda_drugs[['Name', 'SMILES']]
            name_col = 'Name'
            smiles_col = 'SMILES'
                
            print(f"🧹 清洗完毕！我们得到了 {len(fda_drugs)} 个可以用于虚拟筛选的老药。")
        else:
            print("❌ 数据集列名不匹配，请检查原始数据源结构。需要 Name 和 SMILES 列。")
            return
            
    except Exception as e:
        print(f"❌ 读取本地文件失败: {e}")
        return

    # 3. 特征处理与指纹计算
    print("\n🧬 正在将药物转化为分子指纹 (ECFP4)...")
    valid_features = []
    valid_mols = []
    drug_names = []
    drug_smiles = []
    
    for idx, row in fda_drugs.iterrows():
        smi = row[smiles_col]
        name = row[name_col]
        
        # 强制使用模型原本训练时的维度 (2048)
        fp_array, mol = smiles_to_ecfp4(smi, radius=2, n_bits=expected_features)
        
        if fp_array is not None:
            valid_features.append(fp_array)
            valid_mols.append(mol)
            drug_names.append(name)
            drug_smiles.append(smi)
            
    X_pred = np.array(valid_features)
    print(f"✅ 成功转化 {len(X_pred)} 个分子！跳过了 {len(fda_drugs) - len(X_pred)} 个无法解析的畸形结构。")

    # 4. 模型预测打分 (核心步骤)
    print("\n🔮 正在让随机森林评估每一个药物成为高活性 HDAC6 抑制剂的概率...")
    
    # predict_proba 返回两列：[变成0的概率, 变成1的概率]。我们取第2列 (索引1)
    probabilities = best_model.predict_proba(X_pred)[:, 1]
    
    # 5. 结果整理与排序
    results_df = pd.DataFrame({
        'Drug_Name': drug_names,
        'SMILES': drug_smiles,
        'HDAC6_Activity_Prob': probabilities
    })
    
    # 按概率从高到低排序，并根据药物名称进行去重（保留同名药物中预测分数最高的一条记录）
    results_df = results_df.sort_values(by='HDAC6_Activity_Prob', ascending=False)
    results_df = results_df.drop_duplicates(subset=['Drug_Name'], keep='first').reset_index(drop=True)
    
    # 保存完整的预测打分结果
    csv_path = os.path.join(output_dir, "FDA_Repurposing_Hits.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 去重后的预测打分结果（共 {len(results_df)} 个唯一药物）已保存至: {csv_path}")
    
    # 提取排名前 20 的潜力候选药物
    top_n = 20
    top_candidates = results_df.head(top_n)
    
    print(f"\n🏆 ======== 排名前 {top_n} 的老药新用候选分子 (Top Candidates) ========")
    for idx, row in top_candidates.iterrows():
        print(f"Top {idx+1}: {row['Drug_Name']} -> 预测打分: {row['HDAC6_Activity_Prob']:.4f}")
        
    print("==========================================================================")
    
    # 6. 绘图：将其 2D 结构图画在一个网格里供论文展示
    try:
        print("\n🎨 正在绘制 Top 20 药物候选分子的 2D 并排结构图...")
        draw_mols = []
        draw_legends = []
        
        # 重新从 SMILES 提取前 20 的 Mol 对象画图，保证顺序对应
        for idx, row in top_candidates.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol:
                draw_mols.append(mol)
                title = str(row['Drug_Name'])
                # 如果名字太长，截断一下以便显示画图图注
                if len(title) > 15:
                    title = title[:12] + "..."
                draw_legends.append(f"{title}\nProb:{row['HDAC6_Activity_Prob']:.2f}")
                
        img = Draw.MolsToGridImage(
            draw_mols,
            molsPerRow=5,       # 每行放 5 个分子
            subImgSize=(300, 300), 
            legends=draw_legends
        )
        
        plot_path = os.path.join(output_dir, "Top20_Candidates_2D.png")
        img.save(plot_path)
        print(f"✨ 绘图成功！图片已保存至: {plot_path}")
        print("\n🎓 【论文指导提示】")
        print("你可以将这张 Top20 的分子图和打分表放在你论文的最后一章。你可以挑选其中评分最高、血脑屏障透过率好或与你课题最相关的一两个老药进行重点讨论，探讨为什么这台经过 5000 个分子训练的‘AI专家’认为它们有潜力转行去当 HDAC6 抑制剂。这就是典型的‘老药新用’ (Drug Repurposing) 研究套路！")
    
    except Exception as e:
        print(f"❌ 绘制 2D 结构图时出错: {e}")

if __name__ == "__main__":
    main()
