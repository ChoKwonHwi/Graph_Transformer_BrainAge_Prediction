import os
import pandas as pd
import nibabel as nib
import numpy as np

# Nifti file을 npy로 변환하고 저장하는 코드
data_path = '/nasdata4/kwonhwi/GraphTransformer/data/ADNI/MCI'
save_path_DTI = '/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/data/ADNI/MCI/DTI'
save_path_MRI = '/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/data/ADNI/MCI/MRI'

os.makedirs(save_path_DTI, exist_ok=True)
os.makedirs(save_path_MRI, exist_ok=True)

target_shape = (160, 192, 160)

def crop_nonzero_and_resize(arr, target_shape):
    # Non-zero 영역의 좌표 찾기
    non_zero_coords = np.argwhere(arr)
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0) + 1  # max 좌표는 exclusive이므로 +1
    
    # Non-zero 영역만 crop
    cropped_arr = arr[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    
    # cropped array의 shape 가져오기
    cropped_shape = cropped_arr.shape
    
    # 목표 크기와의 차이 계산
    diff = [target_shape[i] - cropped_shape[i] for i in range(3)]
    
    # 중앙에서 자르거나 패딩 추가
    for i in range(3):
        if diff[i] < 0:  # 자를 필요가 있는 경우
            start = (cropped_shape[i] - target_shape[i]) // 2
            cropped_arr = np.take(cropped_arr, indices=range(start, start + target_shape[i]), axis=i)
        elif diff[i] > 0:  # 패딩을 추가해야 하는 경우
            pad_width = [(0, 0)] * 3
            pad_width[i] = (diff[i] // 2, diff[i] - diff[i] // 2)
            cropped_arr = np.pad(cropped_arr, pad_width=pad_width, mode='constant', constant_values=0)
    
    return cropped_arr

sub_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(data_path, sub_dir)
    
    # 파일 경로 설정
    file_MRI_path = os.path.join(sub_dir_path, sub_dir+'_linreg_T1_Warped.nii.gz')
    file_DTI_path = os.path.join(sub_dir_path, sub_dir+'_linreg_DTI_Warped.nii.gz')
    
    if os.path.exists(file_DTI_path) and os.path.exists(file_MRI_path):
        # NIfTI 파일 읽기 및 numpy 배열로 변환
        MRI = nib.load(file_MRI_path).get_fdata()
        DTI = nib.load(file_DTI_path).get_fdata()

        # Non-zero 영역만 crop하고 목표 shape로 맞추기
        MRI_cropped = crop_nonzero_and_resize(MRI, target_shape)
        DTI_cropped = crop_nonzero_and_resize(DTI, target_shape)

        # 저장 파일 경로 설정
        save_file_MRI = os.path.join(save_path_MRI, f"{sub_dir}.npy")
        save_file_DTI = os.path.join(save_path_DTI, f"{sub_dir}.npy")
        
        # numpy 배열로 저장
        np.save(save_file_MRI, MRI_cropped)
        np.save(save_file_DTI, DTI_cropped)
        
        del MRI, DTI, MRI_cropped, DTI_cropped
    else:
        print(f"'{sub_dir}' 파일이 존재하지 않습니다.")


# Age csv file을 npy로 변환한 뒤 저장하는 코드
'''
csv_file_path = '/nasdata4/kwonhwi/GraphTransformer/data/MCI_age.csv' 

df = pd.read_csv(csv_file_path)
df['AGE'] = df['AGE'].astype(float)

base_path = '/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/data/ADNI/MCI/Age'
os.makedirs(base_path, exist_ok=True)

for _, row in df.iterrows():
    id_value = row['id']
    age_value = row['AGE']
    save_path = os.path.join(base_path, f"{id_value}.npy")
    
    # age 값을 numpy 배열로 저장
    np.save(save_path, age_value)
print("Finish")
'''