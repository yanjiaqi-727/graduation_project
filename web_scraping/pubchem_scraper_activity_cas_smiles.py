# -*- coding: utf-8 -*-
import csv
import os
import time
import random
import urllib.parse
from playwright.sync_api import sync_playwright, TimeoutError

save_dir = "web_scraping/activity"
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except Exception as e:
        print(f"Warning: Cannot create directory {save_dir}.", e)

csv_path = "ML/virtual_screening_activity_results/FDA_Repurposing_Hits.csv"
txt_path = "web_scraping/pubchem_scraping_activity_smiles_progress.txt"

def read_progress():
    processed = set()
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip('\n').split('\t')
                if len(parts) > 0:
                    processed.add(parts[0])  # Drug_Name
    return processed

def write_progress(drug_name, smiles, cid, cas, prob, is_2d_success, note):
    file_exists = os.path.exists(txt_path)
    with open(txt_path, 'a', encoding='utf-8') as f:
        if not file_exists:
            f.write("Drug_Name\tSMILES\tCID\tCAS\tHDAC6_Activity_Prob\t2D是否下载成功\t备注\n")
        f.write(f"{drug_name}\t{smiles}\t{cid}\t{cas}\t{prob}\t{is_2d_success}\t{note}\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"    -> Logged to TXT: CAS={cas}, 2D={is_2d_success}")

def main():
    processed_drugs = read_progress()
    drugs_to_process = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row['HDAC6_Activity_Prob']) > 0.5:
                if row['Drug_Name'] not in processed_drugs:
                    drugs_to_process.append(row)
    
    print(f"Total active drugs to process (> 0.5): {len(drugs_to_process)}")
    
    if not drugs_to_process:
        print("All >0.5 molecules processed!")
        return
        
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        
        for drug_info in drugs_to_process:
            drug_name = drug_info['Drug_Name']
            smiles = drug_info['SMILES']
            prob = drug_info['HDAC6_Activity_Prob']
            print(f"\n[{time.strftime('%H:%M:%S')}] Processing: {drug_name} with SMILES")
            
            cid = "未知"
            cas = "未知"
            is_2d_success = "否"
            note = ""
            
            try:
                encoded_smiles = urllib.parse.quote(smiles)
                page.goto(f"https://pubchem.ncbi.nlm.nih.gov/#query={encoded_smiles}", wait_until="domcontentloaded", timeout=60000)
                time.sleep(3)
                
                # Check if it redirected to the compound page
                if "/compound/" in page.url:
                    # Exact match, already on the compound page
                    pass
                else:
                    # Not redirected immediately, wait for results UI
                    try:
                        page.wait_for_selector('span.font-weight-medium, span.p-sm-right.p-sm-left, div.font-medium.p-1.lg\\:flex-1', timeout=15000)
                    except TimeoutError:
                        pass
                        
                    # Check result count text
                    result_text = page.evaluate('''() => {
                        let resultDiv = document.querySelector('div.font-medium.p-1.lg\\\\:flex-1');
                        if (resultDiv && resultDiv.innerText) {
                            return resultDiv.innerText.trim().toLowerCase();
                        }
                        let elements = document.querySelectorAll('div, span');
                        for(let el of elements) {
                            let t = el.innerText ? el.innerText.trim().toLowerCase() : "";
                            if(t === '1 result' || t.endsWith(' results')) {
                                return t;
                            }
                        }
                        return "";
                    }''')
                    
                    if "1 result" == result_text.lower() or "1 result" in result_text.lower():
                        # Exactly 1 result found
                        try:
                            # 直接提取化合物的详情页链接并跳转，避免点击带来的问题
                            target_href = page.evaluate('''() => {
                                let link = document.querySelector('a.regular-link[href*="/compound/"]');
                                if (link) return link.href;
                                
                                let links = document.querySelectorAll('a[href*="/compound/"]');
                                for (let a of links) {
                                    if (a.innerText && a.innerText.trim().length > 0) return a.href;
                                }
                                return "";
                            }''')
                            
                            if target_href:
                                page.goto(target_href, wait_until="domcontentloaded", timeout=60000)
                            else:
                                note = "有 1 result，但未找到跳转链接"
                                write_progress(drug_name, smiles, cid, cas, prob, is_2d_success, note)
                                continue
                        except Exception as e:
                            note = "有 1 result，但跳转详情页失败"
                            write_progress(drug_name, smiles, cid, cas, prob, is_2d_success, note)
                            continue
                    else:
                        note = f"搜索结果不唯一或为空 ({result_text})，已跳过"
                        write_progress(drug_name, smiles, cid, cas, prob, is_2d_success, note)
                        continue
                
                # 等待详情页加载
                try:
                    page.wait_for_selector("#Structures", timeout=20000)
                    time.sleep(2)
                except TimeoutError:
                    note = "进入详情页后，未加载出结构模块"
                    write_progress(drug_name, smiles, cid, cas, prob, is_2d_success, note)
                    continue
                
                # --- 提取 CID ---
                try:
                    url = page.url
                    if "/compound/" in url:
                        cid_from_url = url.split("/compound/")[1].split("#")[0]
                        if cid_from_url.isdigit():
                            cid = cid_from_url
                            
                    if cid == "未知":
                        cid_text = page.evaluate('''() => {
                            const ths = document.querySelectorAll("th");
                            for (let th of ths) {
                                if (th.innerText.includes("PubChem CID")) {
                                    return th.nextElementSibling.innerText.trim();
                                }
                            }
                            return "";
                        }''')
                        if cid_text:
                            cid = cid_text
                except Exception:
                    pass
                
                # --- 提取 CAS ---
                try:
                    cas_text = page.evaluate('''() => {
                        const links = document.querySelectorAll('a[data-ga-action="content-link"]');
                        for (let link of links) {
                            let text = link.innerText.trim();
                            if (text.match(/^\\d{2,7}-\\d{2}-\\d$/)) {
                                return text;
                            }
                        }
                        const allDivs = document.querySelectorAll('div');
                        for (let div of allDivs) {
                            if (div.innerText && div.innerText.includes('CAS')) {
                                let match = div.innerText.match(/\\b(\\d{2,7}-\\d{2}-\\d)\\b/);
                                if (match && match[1]) {
                                    return match[1];
                                }
                            }
                        }
                        return "未知";
                    }''')
                    if cas_text and cas_text != "未知":
                        cas = cas_text
                except Exception as e:
                    pass
                
                if cas == "未知":
                    note += "未匹配到有效的CAS号格式 "
                
                # --- 2D Download ---
                try:
                    section_2d = page.locator('section[id="2D-Structure"]')
                    if section_2d.is_visible():
                        download_btn = section_2d.locator("button:has-text('Download Coordinates')").first
                    else:
                        download_btn = page.locator("#Structures button:has-text('Download Coordinates')").first
                        
                    if download_btn.is_visible():
                        download_btn.scroll_into_view_if_needed()
                        download_btn.click()
                        time.sleep(1)
                        
                        with page.expect_download(timeout=15000) as download_info:
                            save_link = page.locator('a[data-ga-label*="SDF - Save"]').first
                            save_link.click()
                        
                        download = download_info.value
                        
                        # 安全处理文件名：DrugName_CAS.sdf
                        safe_drug_name = "".join([c for c in drug_name if c.isalpha() or c.isdigit() or c in ('_', '-', '.')]).replace(" ", "_")
                        safe_cas = "".join([c for c in cas if c.isdigit() or c == '-'])
                        if not safe_cas:
                            safe_cas = "未知CAS"
                            
                        filename = f"{safe_drug_name}_{safe_cas}.sdf"
                        save_path = os.path.join(save_dir, filename)
                        
                        download.save_as(save_path)
                        is_2d_success = "是"
                    else:
                        note += "未找到 2D 下载按钮 "
                except Exception as e:
                    note += "下载2D出错 "

                write_progress(drug_name, smiles, cid, cas, prob, is_2d_success, note.strip())
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                print(f"Error processing {drug_name}: {e}")
                note = "运行出错: " + str(e).replace('\n', ' ').replace('\t', ' ')
                write_progress(drug_name, smiles, cid, cas, prob, is_2d_success, note)
                
        browser.close()
        print("Done processing!")

if __name__ == "__main__":
    main()
