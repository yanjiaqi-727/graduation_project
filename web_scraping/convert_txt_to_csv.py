import csv
import os

def convert():
    txt_file = "web_scraping/pubchem_scraping_activity_smiles_progress.txt"
    csv_file = "web_scraping/pubchem_scraping_activity_smiles_progress.csv"

    rows = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_clean = line.strip('\n')
            if not line_clean.strip(): 
                continue
            
            parts = line_clean.split('\t')
            # 正常的一行应该有 7 列，如果不满 7 列说明是上一行报错信息（带回车）的换行部分
            if len(parts) >= 7:
                if len(parts) > 7:
                    parts = parts[:6] + [' '.join(parts[6:])]
                rows.append(parts)
            elif len(rows) > 0:
                # 拼接到上一行的最后一列 (备注) 中
                rows[-1][-1] += ' ' + line_clean

    # 使用 utf-8-sig 编码保存，确保 Excel 打开不乱码
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        
    print(f"Successfully converted TXT to CSV! Processed {len(rows)-1} rows.")
    print(f"Saved to: {csv_file}")

if __name__ == '__main__':
    convert()
