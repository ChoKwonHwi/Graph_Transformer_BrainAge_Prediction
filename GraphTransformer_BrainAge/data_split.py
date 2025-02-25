import os
import glob

# Base 경로 설정
base_path = "/nasdata4/5ug/Graph_data/adni/CN"

# 모든 subject 디렉토리 탐색
subjects = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# 각 subject 디렉토리 처리
for subject in subjects:
    subject_path = os.path.join(base_path, subject)
    
    # *_eddy_dti_FA.nii.gz 파일 검색
    file_pattern = os.path.join(subject_path, "*_eddy_dti_FA.nii.gz")
    matching_files = glob.glob(file_pattern)
    
    if matching_files:
        # 첫 번째 매칭 파일만 처리 (하나만 있다고 가정)
        old_file = matching_files[0]
        new_file = os.path.join(subject_path, f"{subject}_eddy_dti_FA.nii.gz")
        
        # 파일 이름 변경
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} -> {new_file}")
    else:
        print(f"No matching files found in {subject_path}")
