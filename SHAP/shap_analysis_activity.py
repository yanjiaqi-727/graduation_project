# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import shap
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# Windows 终端控制台打印 utf-8 可能会报错，重新配置 stdout
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 配置 matplotlib 支持中文显示和高清输出
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def smiles_to_ecfp4(smiles, radius=2, n_bits=1024):
    """
    将 SMILES 转换为 ECFP4 分子指纹 (1024 bits)
    同时返回 info 用于后续画图
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.int8), None, {}
    
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=info)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, mol, info

def main():
    print("🔬 正在进行基于 SHAP 的构效关系 (SAR) 及特征解释分析...")
    
    model_path = "ML/models/Best_HDAC6_Activity_Model.pkl"
    test_set_path = "ML/models/test_set_activity.csv"
    output_dir = "SHAP/shap_analysis_activity_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 {model_path}，请先运行 model_training.py。")
        return
        
    if not os.path.exists(test_set_path):
        print(f"❌ 错误：找不到测试集文件 {test_set_path}，请先运行 model_training.py。")
        return

    # 1. 加载模型和测试集数据
    print(f"📥 正在加载模型：{model_path}")
    best_model = joblib.load(model_path)
    
    print(f"📥 正在读取独立测试集：{test_set_path}")
    df_test = pd.read_csv(test_set_path)
    print(f"   测试集共 {len(df_test)} 个样本 (Label=1: {df_test['Label'].sum()}, Label=0: {len(df_test) - df_test['Label'].sum()})")
    
    # 分层抽样：SHAP 全局分析使用 200 个代表性样本已足够揭示稳定规律，
    # 同时大幅降低 XGBoost 内部 float32 转换时的内存峰值。
    SHAP_SAMPLE_SIZE = 200
    df_test = df_test.dropna(subset=['Smiles'])
    if len(df_test) > SHAP_SAMPLE_SIZE:
        from sklearn.model_selection import train_test_split
        df_shap, _ = train_test_split(
            df_test, train_size=SHAP_SAMPLE_SIZE,
            stratify=df_test['Label'], random_state=42
        )
        print(f"   ✂️ 已分层抽样 {SHAP_SAMPLE_SIZE} 个样本用于 SHAP 分析 (正负样本比例保持不变)")
    else:
        df_shap = df_test
    valid_smiles = df_shap['Smiles'].tolist()
    
    # 2. 重新提取特征
    X_list = []
    mols_list = []
    bit_infos_list = []
    
    print("🧬 正在提取子集的 ECFP4 指纹...")
    for smi in valid_smiles:
        arr, mol, info = smiles_to_ecfp4(smi)
        X_list.append(arr)
        mols_list.append(mol)
        bit_infos_list.append(info)
        
    X = np.array(X_list)
    feature_names = [f"Morgan_{i}" for i in range(X.shape[1])]
    
    # 3. 计算 SHAP 值
    # 提前转为 float32：避免 XGBoost 内部隐式转换时产生额外内存峰值
    X_shap = X.astype(np.float32)
    print("🧮 正在计算 SHAP 值 (可能需要几分钟，请耐心等待)...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap, check_additivity=False)
    
    # XGBoost 的 TreeExplainer 直接返回形状为 (n_samples, n_features) 的数组
    # 正值表示增加了预测模型对预测为 1（高选择性）的概率（对数几率）
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    elif len(shap_values.shape) > 2:
        shap_values_class1 = shap_values[..., 1]
    else:
        shap_values_class1 = shap_values

    # 4. 可视化：SHAP Summary Plot
    print("📈 正在绘制 SHAP Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values_class1, X_shap, feature_names=feature_names, show=False, max_display=15)
    plt.title('XGBoost-ECFP4 SHAP Explaination For Activity', fontsize=14, pad=20)
    plt.tight_layout()
    shap_summary_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(shap_summary_path)
    plt.close()
    print(f"✅ SHAP 蜂群图已保存至: {shap_summary_path}")

    # 5. 药理学微观解释：将最重要位点映射回 RDKit 化学结构
    print("\n🧪 正在提取最关键的化学片段 (Pharmacophores)...")
    mean_abs_shap = np.mean(np.abs(shap_values_class1), axis=0)
    
    # 提取对活性有最强正负向贡献的特征
    for top_k in [5, 15]:
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
        
        molecules_to_draw = []
        legends = []
        bit_imgs = []
        
        print(f"\n🏆 排名前 {top_k} 的重要 ECFP4 指纹位点:")
        for i, bit in enumerate(top_indices):
            direction = "正向 (增加活性)" if np.mean(shap_values_class1[X[:, bit] == 1, bit]) > 0 else "负向 (降低活性)"
            print(f"   Top {i+1}: Morgan_{bit} (平均 SHAP 绝对贡献度: {mean_abs_shap[bit]:.4f}) - {direction}")
            
            # 寻找对应的真实结构图
            found = False
            bit = int(bit)
            
            for mol_idx, (mol, info) in enumerate(zip(mols_list, bit_infos_list)):
                if mol is not None and bit in info:
                    try:
                        import io
                        from PIL import Image
                        img_data = Draw.DrawMorganBit(mol, bit, info, useSVG=False)
                        if isinstance(img_data, bytes):
                            img = Image.open(io.BytesIO(img_data))
                        elif isinstance(img_data, str):
                            img = Image.open(io.BytesIO(img_data.encode('latin1')))
                        else:
                            img = img_data
                        bit_imgs.append(img)
                        # 仅保留 Morgan_XXX 和 SHAP 值
                        legends.append(f"Morgan_{bit}\nSHAP {mean_abs_shap[bit]:.4f}")
                        found = True
                        break
                    except Exception as e:
                        pass
                    
            if not found:
                print(f"⚠️ 未在抽样数据中找到清晰包含 Morgan_{bit} 的分子展示。")

        if bit_imgs:
            try:
                print(f"\n🎨 正在绘制核心药效团化学结构图 (Top {top_k})...")
                from PIL import Image, ImageDraw, ImageFont
                
                # 单张图片宽高
                width, height = bit_imgs[0].size
                # 右侧文字区域预留宽度
                padding_right = 160
                
                try:
                    font = ImageFont.truetype("arial.ttf", 22)
                except IOError:
                    font = ImageFont.load_default()
                
                # Top 15 使用 3 列排版，避免单列太长
                if top_k == 15:
                    cols = 3
                    rows = (len(bit_imgs) + cols - 1) // cols
                    grid_img = Image.new('RGB', ((width + padding_right) * cols, height * rows), 'white')
                    draw = ImageDraw.Draw(grid_img)
                    
                    for i, img in enumerate(bit_imgs):
                        c = i % cols
                        r = i // cols
                        x_offset = c * (width + padding_right)
                        y_offset = r * height
                        grid_img.paste(img, (x_offset, y_offset))
                        text = legends[i]
                        draw.text((x_offset + width, y_offset + height // 2 - 20), text, fill="black", font=font)
                else:
                    # Top 5 维持单列竖排
                    grid_img = Image.new('RGB', (width + padding_right, height * len(bit_imgs)), 'white')
                    draw = ImageDraw.Draw(grid_img)
                    
                    for i, img in enumerate(bit_imgs):
                        grid_img.paste(img, (0, i * height))
                        text = legends[i]
                        draw.text((width, i * height + height // 2 - 20), text, fill="black", font=font)
                    
                if top_k == 5:
                    sar_plot_path = os.path.join(output_dir, "shap_pharmacophores.png")
                else:
                    sar_plot_path = os.path.join(output_dir, f"shap_pharmacophores_top{top_k}.png")
                    
                grid_img.save(sar_plot_path)
                print(f"🧬 Top {top_k} 核心片段结构图已保存至: {sar_plot_path}")
            except Exception as e:
                print(f"❌ 绘制 Top {top_k} 结构图时出错: {e}")
    print("\n💬 【论文写作提示】")
    print("1. SHAP 蜂群图：展示了每个特征（红点为存在该基团，蓝点为不存在）是将预测推向 HDAC6 高活性还是低活性。")
    print("2. 核心片段结构图：这是论文第五章的灵魂！结合你在图上看到的官能团，分析它们为什么发挥了活性的识别作用。")

if __name__ == "__main__":
    main()
