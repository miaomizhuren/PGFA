"""Combine final paper table CSVs into review-friendly CSV/XLSX backups."""
from pathlib import Path
import pandas as pd
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/saved_results/final_paper_tables_20260913'
TABLES=[
 (1,'Main Comparison','Main Comparison','TABLE_1_main_comparison.csv'),
 (2,'Defense Evaluation','Defense Evaluation','TABLE_2_defense.csv'),
 (3,'Ablation Study','Ablation Study','TABLE_3_ablation.csv'),
 (4,'Mechanism Analysis','Mechanism Analysis','TABLE_4_mechanism.csv'),
 (5,'Attack Duration Sensitivity','Attack Duration Sensitivity','TABLE_5_attack_duration.csv'),
 (6,'Prototype Supervision Strength','Prototype Supervision Strength Sensitivity','TABLE_6_lambda_train.csv'),
 (7,'Cross-Dataset Evaluation','Cross-Dataset Evaluation','TABLE_7_cross_dataset.csv'),
 (8,'Data Heterogeneity','Impact of Non-IID Data','TABLE_8_noniid.csv'),
]
PRIORITY=['table_id','table_name','section_name','original_csv','Dataset','Partition','Attack_Window','Attack_Rounds','Method','Variant','Defense','lambda_train','lambda_delta','CA@350','ASR-0','ASR-60','ASR-120','ASR-170','ASR-180','ASR-200','L50','L30','PA-AUC','PA-AUC_0:170','PA-AUC_0:200','S_target','S_source','M','source_dir','notes']
ALIASES={'dataset':'Dataset','partition':'Partition','attack_window':'Attack_Window','attack_rounds':'Attack_Rounds','method':'Method','defense':'Defense','CA_final':'CA@350','PA_AUC':'PA-AUC','M_target_minus_source':'M','S_target_mean':'S_target','S_source_mean':'S_source'}
CLEAN_COLUMNS=['table_id','table_name','section_name','original_csv','Dataset','Partition','Attack_Window','Attack_Rounds','Method','Variant','Defense','lambda_train','lambda_delta','CA@350','ASR-0','ASR-60','ASR-120','ASR-170','ASR-180','ASR-200','L50','L30','PA-AUC','PA-AUC_0:170','PA-AUC_0:200','round','relative_post_round','clean_acc','asr','S_target','S_source','M','S_target_std','S_source_std','feature_norm_mean','num_samples','source_dir','notes']
CLEAN_COALESCE={
 'Method':['Method','method'], 'Defense':['Defense','defense'],
 'Dataset':['Dataset','dataset'], 'Partition':['Partition','partition'],
 'Attack_Window':['Attack_Window','attack_window'],
 'Attack_Rounds':['Attack_Rounds','attack_rounds'],
 'CA@350':['CA@350','CA_final'],
 'ASR-0':['ASR-0','ASR_post0'], 'ASR-60':['ASR-60','ASR_post60'],
 'ASR-120':['ASR-120','ASR_post120'],
 'ASR-180':['ASR-180','ASR_post180'], 'ASR-200':['ASR-200','ASR_post200'],
 'PA-AUC':['PA-AUC','PA_AUC'],
 'S_target':['S_target','S_target_mean'],
 'S_source':['S_source','S_source_mean'],
 'M':['M','M_target_minus_source'],
}
ATTACK_DURATION_POST170={
 ('60-110','CE-UAP'):(51,15.855555555555556),
 ('60-110','PBU'):(51,20.4),
 ('60-150','CE-UAP'):(91,16.144444444444446),
 ('60-150','PBU'):(91,32.65555555555556),
 ('60-180','CE-UAP'):(121,19.7),
 ('60-180','PBU'):(121,42.766666666666666),
}
def write_xlsx(frame,path):
 """Dependency-free minimal XLSX writer; CSV remains the lossless source."""
 def column_name(index):
  name=''
  while True:
   index,remainder=divmod(index,26); name=chr(65+remainder)+name
   if index==0:return name
   index-=1
 rows=[]
 for values in [frame.columns.tolist()]+frame.astype(object).where(frame.notna(),'').values.tolist():
  cells=''.join(f'<c r="{column_name(i)}{len(rows)+1}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>' for i,v in enumerate(values))
  rows.append(f'<row r="{len(rows)+1}">{cells}</row>')
 sheet='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(rows)+'</sheetData></worksheet>'
 with ZipFile(path,'w',ZIP_DEFLATED) as z:
  z.writestr('[Content_Types].xml','<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
  z.writestr('_rels/.rels','<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
  z.writestr('xl/workbook.xml','<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="All Tables" sheetId="1" r:id="rId1"/></sheets></workbook>')
  z.writestr('xl/_rels/workbook.xml.rels','<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
  z.writestr('xl/worksheets/sheet1.xml',sheet)
def coalesce_columns(frame, names):
 """Return the first non-empty value, retaining values exactly as stored."""
 result=pd.Series(pd.NA,index=frame.index,dtype='object')
 for name in names:
  if name in frame:
   values=frame[name].replace(r'^\s*$',pd.NA,regex=True)
   result=result.combine_first(values)
 return result
def patch_attack_duration_fields(clean):
 """Restore the post-170-only fields for the six duration-sensitivity rows."""
 mask=clean['table_id'].eq(5)
 for (attack_window,method),(attack_rounds,asr_post170) in ATTACK_DURATION_POST170.items():
  row=mask & clean['Attack_Window'].eq(attack_window) & clean['Method'].eq(method)
  if row.sum()!=1:
   raise RuntimeError(f'Expected one Attack Duration row for {attack_window} / {method}')
  clean.loc[row,'Attack_Rounds']=attack_rounds
  clean.loc[row,'ASR-170']=asr_post170
 # Duration sensitivity is evaluated over post-attack 0:170, not 0:200.
 clean['PA-AUC_0:170']=clean['PA-AUC_0:170'].astype('object')
 clean.loc[mask,'PA-AUC_0:170']=clean.loc[mask,'PA-AUC']
 clean.loc[mask,'PA-AUC']=pd.NA
def write_clean_combined(raw_path):
 """Create the human-readable view without altering the raw union-column file."""
 raw=pd.read_csv(raw_path)
 clean=pd.DataFrame(index=raw.index)
 for column in CLEAN_COLUMNS:
  if column in CLEAN_COALESCE:
   clean[column]=coalesce_columns(raw,CLEAN_COALESCE[column])
  elif column in raw:
   clean[column]=raw[column]
  else:
   clean[column]=pd.NA
 patch_attack_duration_fields(clean)
 clean_csv=OUT/'ALL_TABLES_COMBINED_CLEAN.csv'
 clean_xlsx=OUT/'ALL_TABLES_COMBINED_CLEAN.xlsx'
 clean.to_csv(clean_csv,index=False,encoding='utf-8-sig')
 write_xlsx(clean,clean_xlsx)
 (OUT/'ALL_TABLES_COMBINED_CLEAN_README.md').write_text('''# Clean combined final paper tables

- `ALL_TABLES_COMBINED.csv` is the original coarse union-column merge.
- `ALL_TABLES_COMBINED_CLEAN.csv` is the recommended human-readable version, with duplicate column names coalesced and redundant duplicate columns removed.
- Attack Duration `Attack_Rounds` and `ASR-170` are completed from the duration-sensitivity source data; its PA-AUC values are retained as `PA-AUC_0:170`.
- No experimental values were changed and missing results were not filled with zero.
- Use the eight individual LaTeX tables for final thesis typesetting.
''',encoding='utf-8')
 return clean_csv,clean_xlsx
def main():
 frames=[]
 for ident,name,section,filename in TABLES:
  path=OUT/filename
  if not path.exists(): raise FileNotFoundError(path)
  frame=pd.read_csv(path)
  for source,alias in ALIASES.items():
   if alias not in frame and source in frame: frame[alias]=frame[source]
  frame.insert(0,'original_csv',filename);frame.insert(0,'section_name',section);frame.insert(0,'table_name',name);frame.insert(0,'table_id',ident)
  frames.append(frame)
 combined=pd.concat(frames,ignore_index=True,sort=False)
 columns=PRIORITY+[c for c in combined.columns if c not in PRIORITY]
 combined=combined.reindex(columns=columns)
 csv_path=OUT/'ALL_TABLES_COMBINED.csv'; xlsx_path=OUT/'ALL_TABLES_COMBINED.xlsx'
 combined.to_csv(csv_path,index=False,encoding='utf-8-sig')
 write_xlsx(combined,xlsx_path)
 (OUT/'ALL_TABLES_COMBINED_README.md').write_text('''# Combined final paper tables

- `ALL_TABLES_COMBINED.csv` is the vertical union-column merge of the eight thesis-table CSVs.
- `ALL_TABLES_COMBINED.xlsx` is the Excel-viewable version.
- Tables have different schemas; unavailable fields remain empty.
- The original eight CSVs were not modified.
- This file is for manual checking, thesis writing, and unified backup; use the eight individual LaTeX tables for final typesetting.
''',encoding='utf-8')
 clean_csv,_=write_clean_combined(csv_path)
 print(f'Combined {len(TABLES)} CSV files into {csv_path}')
 print(f'Created clean combined table at {clean_csv}')
if __name__=='__main__': main()
