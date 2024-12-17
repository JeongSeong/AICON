'''
fsl, freesurfer, dcm2niix가 깔려있어야 함
rapidsai installtion 을 검색하여 additional package에 pytorch, plotly dash, jupyter lab 을 추가선택하여 가상환경을 만들자. 맨 마지막줄은 내가 필요한거 추가한것.
Ex) conda create -n GBM -c rapidsai -c conda-forge -c nvidia  \
    rapids=24.10 python=3.11 'cuda-version>=12.0,<=12.5' \
    jupyterlab 'pytorch=*=*cuda*' dash \
    tensorboard monai feature_engine wandb moviepy lifelines pyclustering trimesh intensity-normalization

conda activate GBM
pip install nnunet
pip install hd_glio  
pip install umap-learn[plot]
conda install jupyter openpyxl
pip install antspyx pyradiomics mrmr_selection

'''
import os
import ants
import numpy as np
import subprocess
import wandb
import pandas as pd
from glob import glob
from monai.transforms import Resize

import SimpleITK as sitk
def make_ADC_of_AMC(patient_dir):
    dwi = sitk.ReadImage(os.path.join(patient_dir, 'dwi.nii.gz'))
    # sitk.WriteImage(dwi[:, :, :, 0], os.path.join(patient_dir, 'dwi_b0.nii.gz'))
    # sitk.WriteImage(dwi[:, :, :, 1], os.path.join(patient_dir, 'dwi_b1000.nii.gz'))
    bval = np.genfromtxt(fname=os.path.join(patient_dir, 'dwi.bval'))
    ADC_array = sitk.GetArrayFromImage(dwi) # z, y, x 저장방향이 바뀜 rpi2rai # nibabel:ras
    ADC = -(np.nan_to_num(np.log(np.nan_to_num(ADC_array[1]/ADC_array[0], nan=0.0, posinf=0.0, neginf=0.0)), nan=0.0, posinf=0.0, neginf=0.0)) / (bval[1]-bval[0]) * 10**6
    dwi0 = dwi[:, :, :, 0] # sitk.ReadImage(os.path.join(patient_dir, 'dwi_b0.nii.gz'))
    pos = sitk.GetImageFromArray(np.where(ADC<0, 0, ADC))
    pos.SetSpacing(dwi0.GetSpacing())
    pos.SetDirection(dwi0.GetDirection())
    pos.SetOrigin(dwi0.GetOrigin())
    sitk.WriteImage(pos, os.path.join(patient_dir, 'adc.nii.gz'))

def dcm2niix_gzip(patient_dir, target_root, subdir:bool):
    if subdir:
        patient_id, patient_date = patient_dir.split('/')[-2:]
        target_dir = os.path.join(target_root, patient_id, patient_date)
    else:
        patient_id = patient_dir.split('/')[-1]
        target_dir = os.path.join(target_root, patient_id)
    os.makedirs(target_dir, exist_ok=True)
    all_data_list = os.listdir(patient_dir)
    for data in all_data_list:
        if '.nii.gz' in data:
            os.system(f'cp -rv "{patient_dir}/{data}" {target_dir}/{data}')
        elif '.nii' in data:
            os.system(f'cp -rv "{patient_dir}/{data}" {target_dir}/{data}')
            os.system(f'gzip {target_dir}/{data}')
        elif data == 'output_DSC':
            dsc_map_path = os.path.join(patient_dir, 'output_DSC')
            for d in os.listdir(dsc_map_path):
                seq_name = d.split(' ')[1]
                if '.nii' in seq_name:
                    target_name = f"{target_dir}/{d}"
                else:
                    target_name = f'{target_dir}/{seq_name}.nii'
                os.system(f'cp -rv "{dsc_map_path}/{d}" "{target_name}"')
                os.system(f'gzip "{target_name}"')
        elif data in ['adc', 'dce', 'dsc', 'dwi', 'flair', 't1', 't1ce', 't2']:
            os.system(f'dcm2niix -z y -f %f -o {target_dir} "{patient_dir}/{data}"')
        else:
            os.system(f'cp -rv "{patient_dir}/{data}" {target_dir}/')

def dcm2niix_report(target_root, subdir:bool): 
    """conversion 결과 csv 파일로 내보내기"""
    # study_dir=[]
    if subdir:
        # for s in seqences:
        #     study_dir.extend(glob(f'{target_root}/*/*/{s}*.nii.gz'))
        study_dir = glob(f'{target_root}/*/*')
    else:
        # for s in seqences:
        #     study_dir.extend(glob(f'{target_root}/*/{s}*.nii.gz'))
        study_dir = glob(f'{target_root}/*')
    def filtering(s_dir, subdir):
        if subdir:
            pid, date = s_dir.split('/')[-2:]
        else:
            pid = s_dir.split('/')[-1]
            date = None
        seqences = ['adc', 'dce', 'dsc', 'dwi', 't1ce', 't1', 't2', 'flair']
        base = {'pid':pid, 'date':date}
        collection=[]
        flag = False
        for s in seqences:
            # collection.extend(glob(f'{s_dir}/{s}*.nii.gz'))
            howMany = len(glob(f'{s_dir}/{s}*.nii.gz'))
            if s == 't1':
                howMany = howMany - base['t1ce']
            if howMany > 1:
                flag=True
            base[s] = howMany
        # base = {'pid':pid, 'date':date, 'adc':0, 'dce':0, 'dsc':0, 'dwi':0, 't1ce':0, 't1':0, 't2':0, 'flair':0}
        if flag:
            return pd.DataFrame(base, index=[0])
        return None

    df_return = pd.DataFrame()
    for directory in study_dir:
        df = filtering(directory, subdir)
        if df is not None:
            df_return = pd.concat([df_return, df], axis=0)
    df_return.to_csv(f'{target_root}/../dcm2niix_report.csv')

def selection(patient_dir, target_root, subdir:bool): # 필요한 파일만 다른 디렉토리에 옮기기
    if subdir:
        patient_id, patient_date = patient_dir.split('/')[-2:]
        target_dir = os.path.join(target_root, patient_id, patient_date)
    else:
        patient_id = patient_dir.split('/')[-1]
        target_dir = os.path.join(target_root, patient_id)
    os.makedirs(target_dir, exist_ok=True)

    try: os.system(f'cp {patient_dir}/t1.nii.gz {target_dir}/t1.nii.gz')
    except Exception as e: print(e)

    try: os.system(f'cp {patient_dir}/t1ce.nii.gz {target_dir}/t1ce.nii.gz')
    except Exception as e: print(e)

    try: os.system(f'cp {patient_dir}/t2.nii.gz {target_dir}/t2.nii.gz')
    except Exception as e: print(e)

    try: os.system(f'cp {patient_dir}/flair.nii.gz {target_dir}/flair.nii.gz')
    except Exception as e: print(e)

    try: os.system(f'cp {patient_dir}/adc.nii.gz {target_dir}/adc.nii.gz')
    except Exception as e: print(e)

    try: os.system(f'cp {patient_dir}/dwi.nii.gz {target_dir}/dwi.nii.gz')
    except Exception as e: print(e)

    try: os.system(f'cp {patient_dir}/K12.nii.gz {target_dir}/Ktrans.nii.gz')
    except Exception as e: print(e)

    try: os.system(f'cp {patient_dir}/dce.nii.gz {target_dir}/dce.nii.gz')
    except Exception as e: print(e)

    try: os.system(f'cp {patient_dir}/rCBV.nii.gz {target_dir}/rCBV.nii.gz')
    except Exception as e: print(e)

    try: os.system(f'cp {patient_dir}/dsc.nii.gz {target_dir}/dsc.nii.gz')
    except Exception as e: print(e)


def bet_MRI_also4D_resume(mri_path): # mri_synthstrip has gpu option: -g
    mri_bet_path = mri_path.replace('.nii.gz', '_bet.nii.gz')
    mri_bet_mask_path = mri_path.replace('.nii.gz', '_bet_mask.nii.gz')
    if not (os.path.exists(mri_bet_path) and os.path.exists(mri_bet_mask_path)):
      mri_type = mri_path.split('/')[-1].split('.nii.gz')[0]
      if mri_type in ['dsc', 'dce', 'dwi']:
          nifti_mri = ants.image_read(mri_path)
          mri_vol_path = mri_path.replace('.nii.gz', '_vol.nii.gz')
          if len(nifti_mri.shape) == 4:
            nifti_mri = ants.slice_image(nifti_mri, axis=3, idx=0)
          nifti_mri.to_file(mri_vol_path)        
          if mri_type == 'dwi':
            os.system(f'fslreorient2std {mri_vol_path} {mri_vol_path}')
          os.system(f'mri_synthstrip -i {mri_vol_path} -o {mri_bet_path} -m {mri_bet_mask_path}')

      else:
          os.system(f'mri_synthstrip -i {mri_path} -o {mri_bet_path} -m {mri_bet_mask_path}')

def ensure_positive(nifti_path): # mri_synthstrip sometimes make the background intensities as negative, which cause malfunction in regiatration.
    min_max = subprocess.run(['fslstats', nifti_path, '-R'], capture_output=True).stdout
    min_max = float(min_max.decode().split(' ')[0])
    if min_max<0:
        # print(nifti_path)
        os.system(f'fslmaths {nifti_path} -thr 0 {nifti_path}')

def check_map(stripped_patient_dir, modal):
    # modals = ["t1ce", "t1", "t2", "adc", "rCBV", "flair", "Ktrans"]
    bright_factor={"t1ce":1, "t1":1, "t2":1, "adc":1, "rCBV":2, "flair":1, "Ktrans":5}
    img = ants.image_read(os.path.join(stripped_patient_dir, f'{modal}_bet.nii.gz')).numpy()
    lower_bound = np.percentile(img, 0.1)
    upper_bound = np.percentile(img, 99.9)
    img = np.clip(img, lower_bound, upper_bound)  # 범위 제한
    img = (img - lower_bound) / (upper_bound - lower_bound)  # 정규화
    img = img*255
    img = np.clip(img*bright_factor[modal], 0, 255)
    img = np.repeat(img[..., np.newaxis], 3, -1) # HWN3 
    # print('each image', img.shape) # (185, 256, 256, 3)
    img = np.transpose(img, (2, 3, 0, 1)) # (N 3 H W)
    zero_slices = np.where(np.all(img == 0, axis=(1, 2, 3)))[0]
    img = np.delete(img, zero_slices, axis=0)
    name = stripped_patient_dir.split('/')[-2:]
    wandb.log({'/'.join(name): wandb.Video(img, fps=4)})

'''
jhlee code refacotred by chatGPT and confirmed by syjeong
'''

class MRIRegistration:
    def __init__(self, target_root, target_resolution, subdir, fixed_image_type='t1', moving_image_types=None, postfix_dict=None):
        self.target_root = target_root
        self.target_resolution = target_resolution
        self.subdir = subdir
        self.fixed_image_type = fixed_image_type
        self.moving_image_types = moving_image_types if moving_image_types is not None else ['t1ce', 't2', 'flair']
        '''
        skull stripping must be already applied on every MRI modals, and saved as follwing names:
        '''
        # Postfix 설정 (필요에 따라 다르게 설정 가능)
        self.postfix = postfix_dict if postfix_dict else {
            'bet': '_bet',
            'mask': '_bet_mask',
            'nifti': '.nii.gz'
        }

    def _get_path(self, patient_dir, image_type, postfix=''):
        """이미지 타입과 postfix로 경로 생성"""
        return os.path.join(patient_dir, f'{image_type}{postfix}{self.postfix["nifti"]}')

    def _get_target_path(self, patient_dir, image_type, postfix=''):
        """타겟 디렉토리에서 저장할 경로 생성"""
        if self.subdir:
            patient_id, study_date = patient_dir.split('/')[-2:] 
            target_dir = os.path.join(self.target_root, patient_id, study_date)
        else:
            patient_id = patient_dir.split('/')[-1]
            target_dir = os.path.join(self.target_root, patient_id)
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f'{image_type}{postfix}{self.postfix["nifti"]}')

    def register_mri(self, fixed_image_path, moving_image_path, bet_moving_image_path, brain_mask_path, transform_type='Rigid', reorient='RPI'):
        """기준 이미지와 이동할 이미지를 받아 Registration 처리"""
        try:
            fixed_img = ants.image_read(fixed_image_path, reorient=reorient)
            moving_img = ants.image_read(moving_image_path, reorient=reorient)
            bet_moving_img = ants.image_read(bet_moving_image_path, reorient=reorient)
            brain_mask = ants.image_read(brain_mask_path, reorient=reorient)

            # 기준 이미지와 이동 이미지를 rigid registration
            bet_moving_img_reg = ants.registration(fixed=fixed_img, moving=bet_moving_img, type_of_transform=transform_type)
            if len(moving_img.shape)==4:
                registered_moving_img = ants.apply_transforms(fixed=bet_moving_img_reg['warpedmovout'], moving=moving_img, transformlist=bet_moving_img_reg['fwdtransforms'], interpolator='hammingWindowedSinc', imagetype=3)
                registered_moving_img_bet = ants.apply_transforms(fixed=bet_moving_img_reg['warpedmovout'], moving=bet_moving_img, transformlist=bet_moving_img_reg['fwdtransforms'])
            else:
                registered_moving_img = ants.apply_transforms(fixed=bet_moving_img_reg['warpedmovout'], moving=moving_img, transformlist=bet_moving_img_reg['fwdtransforms'])
                registered_moving_img_bet = ants.mask_image(registered_moving_img, brain_mask) # Brain mask 적용

            return registered_moving_img, registered_moving_img_bet, bet_moving_img_reg

        except Exception as e:
            print(f"Error during registration: {e}")
            return None, None, None

    def process_fixed_image(self, patient_dir, image_type, reorient='RPI'):
        """기준 이미지를 resample 및 brain extraction mask 생성"""
        try:
            image_path = self._get_path(patient_dir, image_type)
            brain_mask_path = self._get_path(patient_dir, image_type, postfix=self.postfix['mask'])

            img = ants.image_read(image_path, reorient=reorient)
            brain_mask = ants.image_read(brain_mask_path, reorient=reorient)
            print(f'start resampling fixed image: original spacing: {img.spacing}')
            resampled_img = ants.resample_image(img, self.target_resolution, use_voxels=False, interp_type=3)
            resampled_mask = ants.resample_image_to_target(brain_mask, resampled_img, interp_type=1)

            bet_img = ants.mask_image(resampled_img, resampled_mask)

            # 결과 저장
            resampled_img.to_file(self._get_target_path(patient_dir, image_type))
            bet_img.to_file(self._get_target_path(patient_dir, image_type, postfix=self.postfix['bet']))
            resampled_mask.to_file(self._get_target_path(patient_dir, image_type, postfix=self.postfix['mask']))

            return resampled_img, bet_img, resampled_mask

        except Exception as e:
            print(f"Error processing fixed image '{image_type}' for patient '{patient_dir}': {e}")
            return None, None, None

    def process_moving_images(self, patient_dir, fixed_image_type, moving_image_types, reorient='RPI'):
        """여러 이동 이미지를 등록하고 저장"""
        print(f'process fixed image in {patient_dir.split("/")[-1]}')
        _, bet_img, brain_mask = self.process_fixed_image(patient_dir, fixed_image_type, reorient)

        if bet_img is None:
            print(f"Failed to process fixed image '{fixed_image_type}' for patient '{patient_dir}'. Skipping moving images.")
            return
        del _, bet_img, brain_mask

        for moving_type in moving_image_types:
            moving_image_path = self._get_path(patient_dir, moving_type)
            bet_moving_image_path = self._get_path(patient_dir, moving_type, postfix=self.postfix['bet'])

            registered_img, registered_bet_img, _ = self.register_mri(
                self._get_target_path(patient_dir, fixed_image_type, postfix=self.postfix['bet']),
                moving_image_path,
                bet_moving_image_path,
                self._get_target_path(patient_dir, fixed_image_type, postfix=self.postfix['mask']), 
                reorient=reorient
            )

            if registered_img is None or registered_bet_img is None:
                print(f"Failed to register moving image '{moving_type}' for patient '{patient_dir}'.")
                continue

            # 결과 저장
            registered_img.to_file(self._get_target_path(patient_dir, moving_type))
            registered_bet_img.to_file(self._get_target_path(patient_dir, moving_type, postfix=self.postfix['bet']))

    def process_dwi_and_adc(self, patient_dir):
        """
        DWI 및 ADC 이미지를 처리
        4D 시퀀스는 map을 registration하는 데 쓰이는 것이므로 
        bet_MRI_also4D 로 나온 4D_bet 만을 registration 해서 저장 
        4D_bet을 fixed_image 에 registration 하는데 쓰인 matrix를 map registration 할 때 사용
        """
        try:
            print('start to process dwi and adc')
            dwi_path = self._get_path(patient_dir, 'dwi')
            adc_path = self._get_path(patient_dir, 'adc')
            dwi_bet_path = self._get_path(patient_dir, 'dwi', postfix=self.postfix['bet'])
            # adc_bet_path = self._get_path(patient_dir, 'adc', postfix=self.postfix['bet'])

            adc_img = ants.image_read(adc_path, reorient='RPI')
            # adc_bet_img = ants.image_read(adc_bet_path, reorient='RPI')
            template_mask = self._get_target_path(patient_dir, self.fixed_image_type, postfix=self.postfix['mask'])
            # DWI 등록
            registered_dwi_img, registered_dwi_bet_img, dwi_to_fixed_transform = self.register_mri(
                self._get_target_path(patient_dir, self.fixed_image_type, postfix=self.postfix['bet']),
                dwi_path,
                dwi_bet_path,
                template_mask
            )

            if (registered_dwi_img is None) or (registered_dwi_bet_img is None):
                print(f"Failed to register DWI image for patient {patient_dir}")
                return

            # ADC 등록
            adc_img = ants.apply_transforms(dwi_to_fixed_transform['warpedmovout'], adc_img, transformlist=dwi_to_fixed_transform['fwdtransforms'])
            # adc_bet_img = ants.apply_transforms(dwi_to_fixed_transform['warpedmovout'], adc_bet_img, transformlist=dwi_to_fixed_transform['fwdtransforms'])
            adc_bet_img = ants.mask_image(adc_img, ants.image_read(template_mask, reorient='RPI'))

            # 결과 저장
            registered_dwi_img.to_file(self._get_target_path(patient_dir, 'dwi'))
            registered_dwi_bet_img.to_file(self._get_target_path(patient_dir, 'dwi', postfix=self.postfix['bet']))
            adc_img.to_file(self._get_target_path(patient_dir, 'adc'))
            adc_bet_img.to_file(self._get_target_path(patient_dir, 'adc', postfix=self.postfix['bet']))

        except Exception as e:
            print(f"Error processing DWI and ADC for patient {patient_dir}: {e}")

    def process_dsc_and_rCBV(self, patient_dir):
        """
        DSC 및 rCBV 이미지를 처리
        4D 시퀀스는 map을 registration하는 데 쓰이는 것이므로 
        bet_MRI_also4D 로 나온 4D_bet 만을 registration 해서 저장 
        4D_bet을 fixed_image 에 registration 하는데 쓰인 matrix를 map registration 할 때 사용
        """
        try:
            print('start to process dsc and rCBV')
            dsc_path = self._get_path(patient_dir, 'dsc')
            dsc_bet_path =  self._get_path(patient_dir, 'dsc', postfix=self.postfix['bet'])
            cbv_path = self._get_path(patient_dir, 'rCBV')
            # cbv_bet_path = self._get_path(patient_dir, 'rCBV', postfix=self.postfix['bet'])

            cbv_img = ants.image_read(cbv_path, reorient='RPI')
            # cbv_bet_img = ants.image_read(cbv_bet_path, reorient='RPI')
            template_mask = self._get_target_path(patient_dir, self.fixed_image_type, postfix=self.postfix['mask'])

            registered_dsc_img, registered_dsc_bet_img, dsc_to_fixed_transform = self.register_mri(
                self._get_target_path(patient_dir, self.fixed_image_type, postfix=self.postfix['bet']),
                dsc_path,
                dsc_bet_path,
                template_mask
            )

            if (registered_dsc_img is None) or (registered_dsc_bet_img is None):
                print(f"Failed to register DSC image for patient {patient_dir}")
                return
            
            # rCBV 등록
            cbv_img = ants.apply_transforms(dsc_to_fixed_transform['warpedmovout'], cbv_img, transformlist=dsc_to_fixed_transform['fwdtransforms'])
            # cbv_bet_img = ants.apply_transforms(dsc_to_fixed_transform['warpedmovout'], cbv_bet_img, transformlist=dsc_to_fixed_transform['fwdtransforms'])
            cbv_bet_img = ants.mask_image(cbv_img, ants.image_read(template_mask, reorient='RPI'))

            # 결과 저장
            registered_dsc_img.to_file(self._get_target_path(patient_dir, 'dsc'))
            registered_dsc_bet_img.to_file(self._get_target_path(patient_dir, 'dsc', postfix=self.postfix['bet']))
            cbv_img.to_file(self._get_target_path(patient_dir, 'rCBV'))
            cbv_bet_img.to_file(self._get_target_path(patient_dir, 'rCBV', postfix=self.postfix['bet']))

        except Exception as e:
            print(f'Error processing DSC and rCBV for patient {patient_dir}: {e}')
    
    def process_dce_and_Ktrans(self, patient_dir):
        """
        DCE 및 Ktrans 이미지를 처리
        4D 시퀀스는 map을 registration하는 데 쓰이는 것이므로 
        bet_MRI_also4D 로 나온 4D_bet 만을 registration 해서 저장 
        4D_bet을 fixed_image 에 registration 하는데 쓰인 matrix를 map registration 할 때 사용
        """
        try:
            print('start to process dce and Ktrans')
            dce_path = self._get_path(patient_dir, 'dce')
            dce_bet_path =  self._get_path(patient_dir, 'dce', postfix=self.postfix['bet'])
            k_path = self._get_path(patient_dir, 'Ktrans')
            # k_bet_path = self._get_path(patient_dir, 'Ktrans', postfix=self.postfix['bet'])

            k_img = ants.image_read(k_path, reorient='RPI')
            # k_bet_img = ants.image_read(k_bet_path, reorient='RPI')
            template_mask = self._get_target_path(patient_dir, self.fixed_image_type, postfix=self.postfix['mask'])

            registered_dce_img, registered_dce_bet_img, dce_to_fixed_transform = self.register_mri(
                self._get_target_path(patient_dir, self.fixed_image_type, postfix=self.postfix['bet']),
                dce_path,
                dce_bet_path,
                template_mask
            )

            if (registered_dce_img is None) or (registered_dce_bet_img is None):
                print(f"Failed to register DSC image for patient {patient_dir}")
                return
            
            # rCBV 등록
            k_img = ants.apply_transforms(dce_to_fixed_transform['warpedmovout'], k_img, transformlist=dce_to_fixed_transform['fwdtransforms'])
            # k_bet_img = ants.apply_transforms(dsc_to_fixed_transform['warpedmovout'], k_bet_img, transformlist=dsc_to_fixed_transform['fwdtransforms'])
            k_bet_img = ants.mask_image(k_img, ants.image_read(template_mask, reorient='RPI'))

            # 결과 저장
            registered_dce_img.to_file(self._get_target_path(patient_dir, 'dce'))
            registered_dce_bet_img.to_file(self._get_target_path(patient_dir, 'dce', postfix=self.postfix['bet']))
            k_img.to_file(self._get_target_path(patient_dir, 'Ktrans'))
            k_bet_img.to_file(self._get_target_path(patient_dir, 'Ktrans', postfix=self.postfix['bet']))

        except Exception as e:
            print(f"Error processing DCE and Ktrans for patient {patient_dir}: {e}")


    def run_full_registration(self, patient_dir):
        """기본 MRI 이미지와 추가 이미지를 처리"""
        # 기준 이미지와 이동 이미지를 처리
        print(patient_dir)
        self.process_moving_images(patient_dir, fixed_image_type=self.fixed_image_type, moving_image_types=self.moving_image_types)

        # DWI 및 ADC 처리
        self.process_dwi_and_adc(patient_dir)

        # DSC 및 rCBV 처리
        self.process_dsc_and_rCBV(patient_dir)

        # DCE 및 Ktrans 처리
        self.process_dce_and_Ktrans(patient_dir)


def segmentation_mri(patient_dir):
    """Tumor mask 뽑는 함수"""
    os.system(f'hd_glio_predict -t1 {os.path.join(patient_dir, "t1_bet.nii.gz")} -t1c {os.path.join(patient_dir, "t1ce_bet.nii.gz")} -t2 {os.path.join(patient_dir, "t2_bet.nii.gz")} -flair {os.path.join(patient_dir, "flair_bet.nii.gz")} -o {os.path.join(patient_dir, "tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(patient_dir, "tumor_mask.nii.gz")} -mas {os.path.join(patient_dir, "t1ce_bet_mask.nii.gz")} {os.path.join(patient_dir, "tumor_mask.nii.gz")}') # "brain_mask.nii.gz"
    os.system(f'fslmaths {os.path.join(patient_dir, "tumor_mask.nii.gz")} -thr 1 -uthr 1 -bin {os.path.join(patient_dir, "NE_tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(patient_dir, "tumor_mask.nii.gz")} -thr 2 -uthr 2 -bin {os.path.join(patient_dir, "CE_tumor_mask.nii.gz")}')
    os.system(f'fslmaths {os.path.join(patient_dir, "tumor_mask.nii.gz")} -bin {os.path.join(patient_dir, "WT_tumor_mask.nii.gz")}')

def volume_ml_csv(target_root, subdir:bool): # 맨 마지막 슬래쉬 빼고 넣어야 한다.
    def calc_volume_ml(patient_dir): # study_dir
        try:
            ne = subprocess.run(["fslstats", f"{patient_dir}/NE_tumor_mask.nii.gz", "-V"], capture_output=True).stdout
            ne = float(ne.decode().split(' ')[1])/1000
        except Exception as e:
            print(patient_dir)
            print(e)
            ne=None

        try:
            et = subprocess.run(["fslstats", f"{patient_dir}/CE_tumor_mask.nii.gz", "-V"], capture_output=True).stdout
            et = float(et.decode().split(' ')[1])/1000
        except Exception as e:
            print(patient_dir)
            print(e)
            et=None

        try:
            wt = subprocess.run(["fslstats", f"{patient_dir}/WT_tumor_mask.nii.gz", "-V"], capture_output=True).stdout
            wt = float(wt.decode().split(' ')[1])/1000
        except Exception as e:
            print(patient_dir)
            print(e)
            wt=None

        return pd.DataFrame({'study_path':patient_dir, 'et_volme_ml': et, 'ne_volume_ml': ne, 'wt_volume_ml':wt}, index=[0])

    if subdir:
        study_dir = glob(f'{target_root}/*/*')
    else:
        study_dir = glob(f'{target_root}/*')
    output = pd.DataFrame()
    for sd in study_dir:
        df = calc_volume_ml(sd)
        output = pd.concat([output, df], axis=0)
    root_name = target_root.split('/')[-1]
    output.to_csv(f'{target_root}/../{root_name}_volume_ml.csv')

def tissue_seg(mri_path): # only working for structural MRI
    """white matter, gray matter, CSF segmentation하는 함수"""
    seq = mri_path.split('/')[-1]
    if ('t2' in seq): # https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FAST.html
        tt=2
        nn=4
    else: # ('t1' in seq) or ('t1ce' in seq) or ('flair' in seq)
        tt=1
        nn=3
    os.system(f'fast -t {tt} -n {nn} --nopve {mri_path}')

def normal_WM_mask(patient_dir):
    header = ants.image_read(os.path.join(patient_dir, "t1_bet_seg.nii.gz"))
    t1 = header.numpy() == 3
    # t1ce = ants.image_read(os.path.join(patient_dir, "t1ce_bet_seg.nii.gz")).numpy() == 3
    # t2 = ants.image_read(os.path.join(patient_dir, "t2_bet_seg.nii.gz")).numpy() == 4 #flair가 얼추 도와줄거임
    # flair = ants.image_read(os.path.join(patient_dir, "flair_bet_seg.nii.gz")).numpy() == 3
    tumor = ants.image_read(os.path.join(patient_dir, "WT_tumor_mask.nii.gz")).numpy() == 0

    # white = ants.from_numpy((t1ce & t1 & flair & tumor).astype(np.float32), origin=header.origin, spacing=header.spacing, direction=header.direction)
    # white = ants.from_numpy((t1ce & t1 & t2 & flair & tumor).astype(np.float32), origin=header.origin, spacing=header.spacing, direction=header.direction)
    # white = ants.from_numpy((t1 & t2 & tumor).astype(np.float32), origin=header.origin, spacing=header.spacing, direction=header.direction)
    white = ants.from_numpy((t1 & tumor).astype(np.float32), origin=header.origin, spacing=header.spacing, direction=header.direction)
    white.to_file(os.path.join(patient_dir, "NWM_mask.nii.gz"))


def modals_and_NWM_masks(patient_dir):
    """Normal White Matter mask 잘 만들어졌는지 modal에 씌워서 wandb에서 보는 함수"""
    color_map = {-1:[0, 0, 0], 0:[0, 255, 0], 1:[0, 0, 255], 2:[255, 0, 0]}
    alpha = 0.5
    modals = ["t1ce", "t1", "t2", "flair"]
    # bright_factor={"t1ce":1, "t1":1, "t2":1, "flair":1}
    array=None
    # normal white matter mask
    img = ants.image_read(os.path.join(patient_dir, "NWM_mask.nii.gz")).numpy().astype(np.uint8)#*255 # (H W N) 
    img = np.repeat(img[..., np.newaxis], 3, -1) # HWN3 
    img = img*color_map[0]
    # img = np.transpose(img, (2, 0, 1)) # (N H W)
    img = np.transpose(img, (2, 3, 0, 1)) # (N 3 H W)
    # zero_slices = np.where(np.all(img == 0, axis=(1, 2)))[0]
    zero_slices = np.where(np.all(img == 0, axis=(1, 2, 3)))[0]
    img = np.delete(img, zero_slices, axis=0)
    # print('mask shape', img.shape) # (256, 185, 256)
    img = list(img)
    img = np.concatenate(img, axis=-1) # (H W*N) # (3 H W*N)
    # print('before repeat mask', img.shape)
    # mask = np.tile(img, reps=[len(modals), 1]) # (H*len(modals) W*N)
    mask = img[:]
    # print('final mask shape', mask.shape) # (740, 65536)

    for m in modals:
        img = ants.image_read(os.path.join(patient_dir, f'{m}_bet.nii.gz')).numpy()
        lower_bound = np.percentile(img, 0.1)
        upper_bound = np.percentile(img, 99.9)
        img = np.clip(img, lower_bound, upper_bound)  # 범위 제한
        img = (img - lower_bound) / (upper_bound - lower_bound)  # 정규화
        img = img*255
        # img = np.clip(img*bright_factor[m], 0, 255)
        img = np.clip(img, 0, 255)
        img = np.repeat(img[..., np.newaxis], 3, -1) # HWN3 
        # print('each image', img.shape) # (185, 256, 256, 3)
        img = np.transpose(img, (2, 3, 0, 1)) # (N 3 H W)
        img = np.delete(img, zero_slices, axis=0)
        img = list(img) # img.flatten() # (3 H W) list
        img = np.concatenate(img, axis=-1) # (3 H W*N)
        if array is None:
            # array = img[:]
            array = mask*alpha + img[:]*(1-alpha)
            array = np.clip(array*(1/alpha), 0, 255)
            array = np.concatenate((img[:], array), axis=-2)
        else:
            # array = np.concatenate((img[:], array), axis=-2) # concat images along H axis
            masked = mask*alpha + img[:]*(1-alpha)
            # masked = np.clip(masked*(1/alpha), 0, 255)
            array = np.concatenate((array, img[:]), axis=-2) # concat images along H axis
            array = np.concatenate((array, masked), axis=-2) # concat images along H axis

        # print('image getting larger along H axis ', array.shape) # (3 H*len(modals) W*N) # (3, 185, 65536) --> (3, 370, 65536) --> (3, 555, 65536) --> (3, 740, 65536)

    
    array = np.transpose(array, (1, 2, 0)).astype(np.uint8) # (HW3)
    # print('final images', array.shape) 
    
    name = patient_dir.split('/')[-2:]
    # wandb.log({pid: wandb.Image(array, masks={'NWN_mask': {"mask_data": mask, "class_labels": {1:"normal white matter"}}})})
    wandb.log({'/'.join(name): wandb.Image(array)})


def normalize(patient_dir, target_root, subdir):
    if subdir:
        patient_id, patient_date = patient_dir.split('/')[-2:]
        target_dir = os.path.join(target_root, patient_id, patient_date)
    else:
        patient_id = patient_dir.split('/')[-1]
        target_dir = os.path.join(target_root, patient_id)
    # dest_dir = patient_dir.split("/")
    # patient = dest_dir[-1]
    # new_dir_name = dest_dir[-2]+"_normalized"
    # dest_dir = "/".join(dest_dir[:-2] + [new_dir_name, patient])
    os.makedirs(target_dir, exist_ok=True)
    modals = ["t1ce", "t1", "t2", "flair", "rCBV", "adc", "Ktrans"]
    nwm_mask = ants.image_read(os.path.join(patient_dir, "NWM_mask.nii.gz")).numpy()
    ce_mask = ants.image_read(os.path.join(patient_dir, 'CE_tumor_mask.nii.gz'))
    ne_mask = ants.image_read(os.path.join(patient_dir, 'NE_tumor_mask.nii.gz'))
    wt_mask = ants.image_read(os.path.join(patient_dir, 'WT_tumor_mask.nii.gz'))

    ce_array = ce_mask.numpy()
    ne_array = ne_mask.numpy()
    wt_array = wt_mask.numpy()

    for m in modals:
        try:
            img = ants.image_read(os.path.join(patient_dir, f'{m}_bet.nii.gz'))
            array = img.numpy()
            normal_white_matter = array[nwm_mask==1]
            normal_white_matter = normal_white_matter[normal_white_matter>0]
            # mean = np.mean(normal_white_matter)
            mean = np.median(normal_white_matter)
            vox = array / mean
            # https://github.com/AIM-Harvard/pyradiomics/issues/401#issuecomment-411103540
            rad = (array-mean) / np.std(normal_white_matter) * 100 
            vox = ants.from_numpy(vox, origin=img.origin, spacing=img.spacing, direction=img.direction)
            rad = ants.from_numpy(rad, origin=img.origin, spacing=img.spacing, direction=img.direction)
            vox.to_file(os.path.join(target_dir, f'{m}_vox.nii.gz'))
            rad.to_file(os.path.join(target_dir, f'{m}_rad.nii.gz'))

            # rCBV나 Ktrans는 빈 복셀이 많다. 이걸로 multiparametric 분석을 하려면 mask에서 시퀀스가 비어있지 않은 복셀만을 남기는게 좋겠다. 
            ce_array = np.where((ce_array>0)&(array>0), ce_array, 0)
            ne_array = np.where((ne_array>0)&(array>0), ne_array, 0)
            wt_array = np.where((wt_array>0)&(array>0), wt_array, 0)
            
        except Exception as e:
            print(e) 
    os.system(f'cp -v {patient_dir}/CE_tumor_mask.nii.gz {target_dir}/CE_tumor_mask.nii.gz')
    os.system(f'cp -v {patient_dir}/NE_tumor_mask.nii.gz {target_dir}/NE_tumor_mask.nii.gz')
    os.system(f'cp -v {patient_dir}/WT_tumor_mask.nii.gz {target_dir}/WT_tumor_mask.nii.gz')
    ce_array = ants.from_numpy(ce_array, origin=ce_mask.origin, spacing=ce_mask.spacing, direction=ce_mask.direction)
    ne_array = ants.from_numpy(ne_array, origin=ne_mask.origin, spacing=ne_mask.spacing, direction=ne_mask.direction)
    wt_array = ants.from_numpy(wt_array, origin=ce_mask.origin, spacing=ce_mask.spacing, direction=ce_mask.direction)
    ce_array.to_file(os.path.join(target_dir, 'CE_tumor_mask_multi.nii.gz'))
    ne_array.to_file(os.path.join(target_dir, 'NE_tumor_mask_multi.nii.gz'))
    wt_array.to_file(os.path.join(target_dir, 'WT_tumor_mask_multi.nii.gz'))


def check_geometry(nifti1_path, nifti2_path):
    nifti1 = ants.image_read(nifti1_path)
    nifti2 = ants.image_read(nifti2_path)
    if np.array_equal(nifti1.origin, nifti2.origin) and np.array_equal(nifti1.spacing, nifti2.spacing) and np.array_equal(nifti1.direction, nifti2.direction) and np.array_equal(nifti1.shape, nifti2.shape):
        return True
    else:
        return False


def plot_all_resized(study_dir, modals, masks, mask_color_map):
    HW = (256, 256)#, 154) # desired image sized showing on the tensorboard
    resize = Resize(HW, anti_aliasing=True)

    # alpha = 0.5
    array=None
    mask = None
    for m, c in zip(masks, mask_color_map):
        img = ants.image_read(os.path.join(study_dir, m)).numpy().astype(np.uint8) # HWN
        img = np.transpose(img, (2, 0, 1)) # (N H W)
        img = resize(img) # (N H W)
        img = np.transpose(img, (1, 2, 0)) # HWN
        img = np.repeat(img[..., np.newaxis], 3, -1) # HWN3 
        img = img*c
        img = np.transpose(img, (2, 3, 0, 1)) # (N 3 H W)
        img = list(img) # img.flatten() # (3 H W) list
        img = np.concatenate(img, axis=-1) # (3 H W*N)
        if mask is None:
            mask = img.copy()
        else:
            mask = mask+img.copy()
            mask = np.clip(mask, 0, 255)

    for m in modals:
        img = ants.image_read(os.path.join(study_dir, f'{m}.nii.gz')).numpy()
        lower_bound = np.percentile(img, 0.1)
        upper_bound = np.percentile(img, 99.9)
        img = np.clip(img, lower_bound, upper_bound)  # 범위 제한
        img = (img - lower_bound) / (upper_bound - lower_bound)  # 정규화
        img = img*255
        # img = np.clip(img*bright_factor[m], 0, 255)
        img = np.clip(img, 0, 255) # HWN
        img = np.transpose(img, (2, 0, 1)) # (N H W)
        img = resize(img) # (N H W)
        img = np.transpose(img, (1, 2, 0)) # HWN
        img = np.repeat(img[..., np.newaxis], 3, -1) # HWN3 
        img = np.transpose(img, (2, 3, 0, 1)) # (N 3 H W)
        img = list(img) # img.flatten() # (3 H W) list
        img = np.concatenate(img, axis=-1) # (3 H W*N)
        if array is None:
            # # array = img[:]
            # array = mask*alpha + img[:]*(1-alpha)
            # array = np.clip(array*(1/alpha), 0, 255)
            array = img.copy()
            array[mask > 0] = mask[mask > 0] 
            array = np.concatenate((img.copy(), array), axis=-2)
        else:
            # # array = np.concatenate((img[:], array), axis=-2) # concat images along H axis
            # masked = mask*alpha + img[:]*(1-alpha)
            # masked = np.clip(masked*(1/alpha), 0, 255)
            masked = img.copy()
            masked[mask > 0] = mask[mask > 0] 
            array = np.concatenate((array, img.copy()), axis=-2) # concat images along H axis
            array = np.concatenate((array, masked), axis=-2) # concat images along H axis
    array = np.transpose(array, (1, 2, 0)).astype(np.uint8) # (HW3)

    # zero_slices = np.where(np.all(img == 0, axis=(1, 2, 3)))[0]
    
    name = study_dir.split('/')[-2:]
    wandb.log({'/'.join(name): wandb.Image(array)})

