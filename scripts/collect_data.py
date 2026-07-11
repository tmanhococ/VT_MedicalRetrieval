import os
import sys
import urllib.request
import json
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Configure paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DICT_DIR = os.path.join(DATA_DIR, "dictionaries")

# URLs
ICD10_URL = "http://bvck27thang2.com/upload/bvck27thang2/files/shares/DM_BAI_VIET_KHOA/CLBV/DM%20ICD10-19_8_BYT.xlsx"
ICD10_FALLBACK_URL = "https://quangtrihospital.vn/uploads/phu_luc_danh_muc_icd_10_ban_hanh_kem_theo_qd_4469_thay_the_danh_muc_dung_chung_phien_ban_so_6.xls"
RXNORM_URL = "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=IN+MIN+BN+PIN+SBD+SCD"

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    req = urllib.request.Request(url, headers=headers)
    import ssl
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(req, context=context) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())
    print("Download completed.")

def collect_icd10():
    os.makedirs(DICT_DIR, exist_ok=True)
    temp_excel_path = os.path.join(DICT_DIR, "temp_icd10.xlsx")
    csv_output_path = os.path.join(DICT_DIR, "icd10_vi.csv")
    
    # Try download and parsing
    try:
        download_file(ICD10_URL, temp_excel_path)
    except Exception as e:
        print(f"Primary URL failed: {e}. Trying fallback...")
        try:
            download_file(ICD10_FALLBACK_URL, temp_excel_path)
        except Exception as fe:
            print(f"Fallback URL failed: {fe}. Cannot retrieve ICD-10.")
            return False
            
    try:
        print("Parsing Excel file...")
        # Load excel file
        xls = pd.ExcelFile(temp_excel_path)
        # Load first sheet
        df = pd.read_excel(xls, 0)
        
        # Try to guess columns
        code_col = None
        desc_col = None
        
        # Look for headers in first few rows or columns
        for col in df.columns:
            col_str = str(col).lower().strip()
            if col_str in ["mã", "ma", "code", "mã bệnh", "mabenh", "ma_benh"]:
                code_col = col
            elif col_str in ["tên", "ten", "description", "tên bệnh", "tenbenh", "ten_benh", "tên tiếng việt", "tentiengviet"]:
                desc_col = col
                
        # If headers were not in column index (e.g. read from middle of sheet), search the first few rows
        if code_col is None or desc_col is None:
            for idx, row in df.head(10).iterrows():
                row_values = [str(x).lower().strip() for x in row.values]
                for c_idx, val in enumerate(row_values):
                    if val in ["mã", "ma", "code", "mã bệnh", "mabenh", "ma_benh"]:
                        code_col = df.columns[c_idx]
                    elif val in ["tên", "ten", "description", "tên bệnh", "tenbenh", "ten_benh", "tên tiếng việt", "tentiengviet", "tên bệnh (tiếng việt)"]:
                        desc_col = df.columns[c_idx]
                        
        # Fallback to column index 0 and 1 (or 0 and 2 etc) if guessing fails
        if code_col is None:
            code_col = df.columns[0]
        if desc_col is None:
            desc_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
        print(f"Detected columns: Code='{code_col}', Description='{desc_col}'")
        
        # Extract and clean
        parsed_df = df[[code_col, desc_col]].copy()
        parsed_df.columns = ["code", "description"]
        parsed_df = parsed_df.dropna()
        parsed_df["code"] = parsed_df["code"].astype(str).str.strip().str.upper()
        parsed_df["description"] = parsed_df["description"].astype(str).str.strip()
        
        # Filter out rows that are column names or headers
        parsed_df = parsed_df[parsed_df["code"].str.match(r"^[A-Z][0-9]")]
        
        # Save to CSV
        parsed_df.to_csv(csv_output_path, index=False, encoding="utf-8")
        print(f"Successfully compiled {len(parsed_df)} ICD-10 records to {csv_output_path}")
        
    except Exception as e:
        print(f"Error parsing ICD-10 excel: {e}")
        return False
    finally:
        # Cleanup temp file
        if os.path.exists(temp_excel_path):
            try:
                os.remove(temp_excel_path)
            except Exception:
                pass
    return True

def collect_rxnorm():
    os.makedirs(DICT_DIR, exist_ok=True)
    temp_json_path = os.path.join(DICT_DIR, "temp_rxnorm.json")
    csv_output_path = os.path.join(DICT_DIR, "rxnorm_concepts.csv")
    
    try:
        download_file(RXNORM_URL, temp_json_path)
        with open(temp_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        concept_list = []
        # Parse RxNav minConceptList
        min_concept_list = data.get("minConceptList", {}).get("minConcept", [])
        for item in min_concept_list:
            concept_list.append({
                "rxcui": item.get("rxcui", "").strip(),
                "name": item.get("name", "").strip(),
                "tty": item.get("tty", "").strip()
            })
            
        if not concept_list:
            # Check other possible layouts
            min_concept_group = data.get("minConceptGroup", {}).get("minConcept", [])
            for item in min_concept_group:
                concept_list.append({
                    "rxcui": item.get("rxcui", "").strip(),
                    "name": item.get("name", "").strip(),
                    "tty": item.get("tty", "").strip()
                })
                
        df = pd.DataFrame(concept_list)
        df = df.dropna()
        df = df[df["rxcui"] != ""]
        df.to_csv(csv_output_path, index=False, encoding="utf-8")
        print(f"Successfully compiled {len(df)} RxNorm concepts to {csv_output_path}")
        
    except Exception as e:
        print(f"Error collecting RxNorm concepts: {e}")
        return False
    finally:
        if os.path.exists(temp_json_path):
            try:
                os.remove(temp_json_path)
            except Exception:
                pass
    return True

def main():
    print("=== GATHERING COMPLETED MEDICAL DATABASES ===")
    icd_ok = collect_icd10()
    rx_ok = collect_rxnorm()
    if icd_ok and rx_ok:
        print("\nAll databases collected and compiled successfully!")
    else:
        print("\nSome databases failed to compile. Check errors above.")

if __name__ == "__main__":
    main()
