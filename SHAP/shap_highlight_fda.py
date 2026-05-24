# -*- coding: utf-8 -*-
"""
SHAP 高权重特征高亮脚本
————————————————
功能：读取 SHAP-LID.xlsx 中的 FDA 候选药物分子，将 SHAP 分析中排名
      Top 15 的 ECFP4 指纹位点映射回每个分子的真实 2D 结构，并以红色
      高亮命中的原子与化学键，导出高清白底 PNG 图片。

输出目录：picture/
命名格式：{ID}_{drug_name}.png
"""

import os
import sys
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

# Windows 终端 UTF-8 兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# ★ Top 15 高权重 ECFP4 指纹位点（来自 SHAP 分析独立测试集）
#   由 shap_analysis_activity.py 计算得出
# ============================================================
TOP15_BITS = {
    607: {"rank": 1,  "shap": 0.5406, "direction": "positive"},
    878: {"rank": 2,  "shap": 0.4828, "direction": "positive"},
    898: {"rank": 3,  "shap": 0.1959, "direction": "positive"},
    119: {"rank": 4,  "shap": 0.1910, "direction": "positive"},
    807: {"rank": 5,  "shap": 0.1334, "direction": "negative"},
    544: {"rank": 6,  "shap": 0.1278, "direction": "positive"},
    238: {"rank": 7,  "shap": 0.1229, "direction": "positive"},
    511: {"rank": 8,  "shap": 0.1161, "direction": "positive"},
    673: {"rank": 9,  "shap": 0.1097, "direction": "positive"},
    843: {"rank": 10, "shap": 0.1062, "direction": "positive"},
    433: {"rank": 11, "shap": 0.0921, "direction": "negative"},
    428: {"rank": 12, "shap": 0.0889, "direction": "positive"},
    831: {"rank": 13, "shap": 0.0860, "direction": "positive"},
    310: {"rank": 14, "shap": 0.0789, "direction": "positive"},
    656: {"rank": 15, "shap": 0.0788, "direction": "positive"},
}

MORGAN_RADIUS = 2
MORGAN_NBITS = 1024
IMG_SIZE = (800, 600)  # 单张图片分辨率


def get_highlighted_atoms_and_bonds(mol, bit_info, target_bits):
    """
    根据 bitInfo 字典，找出目标位点对应的原子索引和化学键索引。
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        分子对象
    bit_info : dict
        Morgan 指纹的 bitInfo，格式 {bit: [(center_atom, radius), ...]}
    target_bits : set
        需要高亮的目标位点集合
    
    Returns
    -------
    hit_atoms : set  命中的原子索引
    hit_bonds : set  命中的化学键索引
    matched_bits : dict  命中的位点 → 对应的原子环境信息
    """
    hit_atoms = set()
    hit_bonds = set()
    matched_bits = {}

    for bit in target_bits:
        if bit not in bit_info:
            continue

        matched_bits[bit] = []

        for center_atom, radius in bit_info[bit]:
            # 收集以 center_atom 为圆心、radius 为拓扑半径的所有原子
            if radius == 0:
                hit_atoms.add(center_atom)
                matched_bits[bit].append((center_atom, radius))
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
                atoms_in_env = set()
                for bond_idx in env:
                    hit_bonds.add(bond_idx)
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms_in_env.add(bond.GetBeginAtomIdx())
                    atoms_in_env.add(bond.GetEndAtomIdx())
                hit_atoms.update(atoms_in_env)
                matched_bits[bit].append((center_atom, radius))

    return hit_atoms, hit_bonds, matched_bits


def draw_highlighted_molecule(mol, hit_atoms, hit_bonds, drug_name, matched_info):
    """
    使用 RDKit 的 MolDraw2DCairo 生成带红色高亮的白底高清 2D 分子图，
    并在每个命中位点的中心原子旁标注 Morgan_XXX 文字。
    
    Returns
    -------
    PIL.Image  生成的图片对象
    """
    from rdkit.Chem import rdCoordGen
    from PIL import ImageDraw, ImageFont
    rdCoordGen.AddCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(IMG_SIZE[0], IMG_SIZE[1])

    # 绘图参数
    opts = drawer.drawOptions()
    opts.setBackgroundColour((1, 1, 1, 1))  # 白底
    opts.bondLineWidth = 2.0
    opts.padding = 0.15

    # 为命中原子和键设置红色高亮
    atom_colors = {a: (0.9, 0.1, 0.1, 0.45) for a in hit_atoms}
    bond_colors = {b: (0.9, 0.1, 0.1, 0.45) for b in hit_bonds}
    atom_radii = {a: 0.35 for a in hit_atoms}

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(hit_atoms),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=list(hit_bonds),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()

    png_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")

    # ---- 收集所有原子的像素坐标，用于避让 ----
    all_atom_positions = []
    for atom_idx in range(mol.GetNumAtoms()):
        apos = drawer.GetDrawCoords(atom_idx)
        all_atom_positions.append((int(apos.x), int(apos.y)))

    # ---- 在图片上叠加 Morgan_XXX 文字标注 ----
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    label_color = (200, 30, 30, 255)        # 红色文字
    line_color = (200, 30, 30, 150)          # 引线颜色
    used_label_rects = []                    # 已用标签区域 [(x1,y1,x2,y2), ...]

    import math

    for bit, envs in matched_info.items():
        if not envs:
            continue

        center_atom, radius = envs[0]
        pos = drawer.GetDrawCoords(center_atom)
        px, py = int(pos.x), int(pos.y)

        label_text = f"Morgan_{bit}"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # ---- 在 12 个方向上搜索最空旷的位置 ----
        best_pos = None
        best_score = -1
        offsets = [60, 80, 100, 120]

        for dist in offsets:
            for angle_deg in range(0, 360, 30):
                angle = math.radians(angle_deg)
                cx = px + int(dist * math.cos(angle)) - tw // 2
                cy = py + int(dist * math.sin(angle)) - th // 2

                # 边界检查
                if cx < 4 or cy < 4 or cx + tw > IMG_SIZE[0] - 4 or cy + th > IMG_SIZE[1] - 4:
                    continue

                label_rect = (cx, cy, cx + tw, cy + th)

                # 与已放置标签的重叠检查
                overlap_label = False
                for (rx1, ry1, rx2, ry2) in used_label_rects:
                    if not (label_rect[2] < rx1 - 6 or label_rect[0] > rx2 + 6 or
                            label_rect[3] < ry1 - 6 or label_rect[1] > ry2 + 6):
                        overlap_label = True
                        break
                if overlap_label:
                    continue

                # 计算与所有原子的最小距离作为得分（越远越好）
                min_dist_to_atom = float('inf')
                center_lx, center_ly = cx + tw // 2, cy + th // 2
                for (ax, ay) in all_atom_positions:
                    d = math.sqrt((center_lx - ax) ** 2 + (center_ly - ay) ** 2)
                    if d < min_dist_to_atom:
                        min_dist_to_atom = d

                if min_dist_to_atom > best_score:
                    best_score = min_dist_to_atom
                    best_pos = (cx, cy)

            if best_pos and best_score > 40:
                break

        if best_pos is None:
            # 兜底：放在右上方
            best_pos = (min(px + 50, IMG_SIZE[0] - tw - 4),
                        max(py - 50, 4))

        lx, ly = best_pos
        used_label_rects.append((lx, ly, lx + tw, ly + th))

        # 画引线：从标签中心到高亮中心原子
        label_cx, label_cy = lx + tw // 2, ly + th // 2
        draw.line([(label_cx, label_cy), (px, py)], fill=line_color, width=1)

        # 画文字（无边框，纯文字）
        draw.text((lx, ly), label_text, fill=label_color, font=font)

    # 合并图层
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return img


def main():
    print("🔬 SHAP 高权重特征高亮分析脚本启动...\n")

    # ---- 1. 读取 SHAP-LID.xlsx ----
    xlsx_path = "SHAP/SHAP-LID.xlsx"
    if not os.path.exists(xlsx_path):
        print(f"❌ 找不到输入文件: {xlsx_path}")
        return

    df = pd.read_excel(xlsx_path)
    print(f"📥 已读取 {len(df)} 个候选药物 (来自 SHAP-LID.xlsx)")
    print(f"   列名: {list(df.columns)}\n")

    # ---- 2. 准备输出目录 ----
    output_dir = "picture"
    os.makedirs(output_dir, exist_ok=True)

    target_bits = set(TOP15_BITS.keys())

    # ---- 3. 逐个分子处理 ----
    for _, row in df.iterrows():
        drug_id = row['ID']
        drug_name = str(row['drug_name']).strip()
        smiles = str(row['SMILES']).strip()

        print(f"━━━ [{drug_id}] {drug_name} ━━━")
        print(f"    SMILES: {smiles}")

        # 3.1 解析分子
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"    ⚠️ 无法解析 SMILES，跳过。\n")
            continue

        # 3.2 计算 Morgan 指纹 + bitInfo
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, MORGAN_RADIUS, nBits=MORGAN_NBITS, bitInfo=bit_info
        )

        # 3.3 与 Top 15 取交集
        mol_active_bits = set(bit_info.keys())
        hit_bits = mol_active_bits & target_bits

        print(f"    该分子激活的指纹位点总数: {len(mol_active_bits)}")
        print(f"    命中 Top 15 高权重位点数: {len(hit_bits)}")

        if hit_bits:
            for bit in sorted(hit_bits, key=lambda b: TOP15_BITS[b]["rank"]):
                info = TOP15_BITS[bit]
                dir_cn = "正向(↑活性)" if info["direction"] == "positive" else "负向(↓活性)"
                print(f"      ✓ Morgan_{bit} (Rank {info['rank']}, "
                      f"SHAP={info['shap']:.4f}, {dir_cn})")
        else:
            print(f"    ⚠️ 该分子未命中任何 Top 15 高权重特征。")

        # 3.4 获取需高亮的原子和键
        hit_atoms, hit_bonds, matched_bits = get_highlighted_atoms_and_bonds(
            mol, bit_info, hit_bits
        )
        print(f"    高亮原子数: {len(hit_atoms)}, 高亮化学键数: {len(hit_bonds)}")

        # 3.5 绘制高亮分子图
        img = draw_highlighted_molecule(mol, hit_atoms, hit_bonds, drug_name, matched_bits)

        # 3.6 保存图片
        safe_name = drug_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        filename = f"{drug_id}_{safe_name}.png"
        save_path = os.path.join(output_dir, filename)
        img.save(save_path, dpi=(300, 300))
        print(f"    ✅ 已保存: {save_path}\n")

    print("🎉 全部完成！所有带 SHAP 高亮标注的分子图已保存至 picture/ 目录。")
    print("📝 【论文提示】这些图可直接用于论文 3.3.5 节的插图排版。")
    print("   红色高亮区域 = 模型关注的关键药效团片段。")


if __name__ == "__main__":
    main()
