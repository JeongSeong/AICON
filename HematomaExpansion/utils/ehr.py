import os
import sys
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import joblib
'''smchou/hematoma/src/utils/prep_ehr.py'''
def dict_list(df, exclude_cols=["id", "image", "label", "mask"]):
    id_col = df["id"]
    image_col = df["image"]
    label_col = df["label"]
    mask_col = df["mask"]
    clinic_cols = df.drop(columns=exclude_cols)
    # 딕셔너리 리스트 생성
    result = []
    for i in range(len(df)):
        result.append(
            {
                "id": id_col.iloc[i],
                "image": image_col.iloc[i],
                "mask": mask_col.iloc[i],
                "label": label_col.iloc[i],
                "clinic": clinic_cols.iloc[i].to_numpy(),
            }
        )
    return result


def scale_data(
    df,
    mode,
    # FTT의 경우, categorical feature를 scaling 안하고 그냥 넣음. 만약 다른 모델을 쓰려면 "e", "v", "m" 제외
    exclude_cols=["id", "image", "label", "mask", "sex", "ivh", "e", "v", "m", "antiplatelet_anticoagulation"],
    scaler=None,
    save_path=None,
):
    if mode not in ["train", "test"]:
        raise ValueError("Invalid mode. Choose either 'train' or 'test'.")

    scaled_df = df.drop(columns=exclude_cols)

    if mode == "train":
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(scaled_df)
        joblib.dump(scaler, os.path.join(save_path, 'scaler.joblib'))
        scaled_df = pd.DataFrame(scaled_values, columns=scaled_df.columns)
        df[scaled_df.columns] = scaled_df
        return df, scaler

    elif mode == "test":
        if scaler is None:
            if save_path is not None:
                scaler = joblib.load(save_path)
            else:
                sys.exit("scaler must be provided in test mode.")
        scaled_values = scaler.transform(scaled_df)
        scaled_df = pd.DataFrame(scaled_values, columns=scaled_df.columns)
        df[scaled_df.columns] = scaled_df
        return df


def impute_data(
    df,
    mode,
    impute_methods="iterative",
    exclude_cols=["id", "image", "label", "mask"],
    imputer=None,
    save_path=None,
):
    if mode not in ["train", "test"]:
        raise ValueError("Invalid mode. Choose either 'train' or 'test'.")

    impute_df = df.drop(columns=exclude_cols)

    if mode == "train":
        if impute_methods == "iterative":
            imputer = IterativeImputer(max_iter=10, random_state=0, tol=1e-4)
        elif impute_methods == "knn":
            imputer = KNNImputer(n_neighbors=3, copy=True)
        else:
            raise ValueError(
                "Invalid impute_methods. Choose either 'iterative' or 'knn'."
            )
        imputer.set_output()
        imputed_values = imputer.fit_transform(impute_df)
        
        joblib.dump(imputer, os.path.join(save_path, "imputer.joblib"))
        imputed_df = pd.DataFrame(imputed_values, columns=impute_df.columns)
        df[impute_df.columns] = imputed_df

        return df, imputer 

    elif mode == "test":
        if imputer is None:
            if save_path is not None:
                imputer = joblib.load(save_path)
            else:
                sys.exit("Imputer must be provided in test mode.")
        
        imputed_values = imputer.transform(impute_df)
        imputed_df = pd.DataFrame(imputed_values, columns=impute_df.columns)
        df[impute_df.columns] = imputed_df

        return df 


def snuh_prep(df_path, image_path, mask_path):
    df = pd.read_csv(df_path)
    # replace columns
    to_replace = {
        "FU (0:no increased, 1: increase, 2: increase안되었으나 수술, 3: 치료안되었거나 전원, 6: no f/u)": "label"
    }
    df.rename(columns=to_replace, inplace=True)
    # drop rows if label is 2
    df = df[df["label"] != 2].reset_index(drop=True)
    clinics = pd.DataFrame(
        {
            "sex": df["Sex"], # M, F
            "age": df["Age"],
            "ivh": df["IVH"], # 1, 0
            "sbp": df["SBP1"],
            "dbp": df["DBP1"],
            "e": df["E"], # 1, 2, 3, 4
            "v": df["V"], # 1, 2, 3, 4, 5
            "m": df["M"], # 1, 2, 3, 4, 5, 6
            "inr": df[" Lab_INR_NEW"],
            "fibrinogen": df[" Lab_fibrinogen"],
            "platet": df["Lab_Platelet"],
            "aptt": df["Lab_aPTT"],
            "bt": df["BT"],
            "antiplatelet_anticoagulation": df["Antiplatelet/Anticoagulation "], # 1, 0
        }
    )#.reset_index(drop=True)

    clinics["sex"] = clinics["sex"].map({"M": 1, "F": 0})
    clinics = clinics.replace(-1, np.nan)
    clinics["dbp"] = clinics["dbp"].map(lambda x: np.nan if x in [0, 1] else x)
    clinics["sbp"] = clinics["sbp"].map(lambda x: np.nan if x in [0, 1] else x)

    # clinics.fillna(0, inplace=True)
    clinics = clinics.apply(lambda x: pd.to_numeric(x, errors="coerce"))
    
    # change dtypes
    clinics["sex"] = clinics["sex"].astype(float)
    clinics["age"] = clinics["age"].astype(float)
    clinics["ivh"] = clinics["ivh"].astype(float)

    # merge ids, nifti_path, label to clinics
    # clinics["id"] = df["PatientID"].astype(str)
    clinics["id"] = df["PatientID"].apply(lambda id: str(id))
    
    clinics["image"] = df['PatientID'].apply(lambda id: os.path.join(image_path, f"{id}.nii.gz"))
    clinics["image"] = clinics["image"].astype(str)
    clinics["mask"] = df['PatientID'].apply(lambda id: os.path.join(mask_path, f"{id}.nii.gz"))
    clinics["mask"] = clinics["mask"].astype(str)
    clinics["label"] = df["label"].astype(int)
    # remove rows s.t. nifti_path does not exist
    missing = clinics.copy()
    clinics = clinics[clinics['image'].apply(os.path.exists)]
    missing = missing[~missing['image'].apply(os.path.exists)]
    print(len(missing), '개 CT가 없다')
    print(missing)
    '''
    49288696
    45105597
    43366093
    39030030
    26001841
    23025381
    9500716
    9321146
    8375133
    7751950
    7701739
    '''
    return clinics


def brm_prep(df_path, image_path, mask_path):
    df = pd.read_csv(df_path)
    # drop rows if label is 2
    df = df[df["label"] != 2].reset_index(drop=True)
    clinics = pd.DataFrame(
        {
            "sex": df["sex"], # M, F
            "age": df["age"],
            "ivh": df["ivh"], # 1, 0
            "sbp": df["sbp"],
            "dbp": df["dbp"],
            "e": df["e"], # 1, 2, 3, 4
            "v": df["v"], # 1, 2, 3, 4, 5
            "m": df["m"], # 1, 2, 3, 4, 5, 6
            "inr": df["inr"],
            "fibrinogen": df["fibrinogen"],
            "platet": df["platet"],
            "aptt": df["aptt"],
            "bt": df["bt"],
            "antiplatelet_anticoagulation": df["antiplatelet_anticoagulation"], # 1, 0
        }
    )#.reset_index(drop=True)
    clinics["sex"] = clinics["sex"].map({"M": 1, "F": 0})
    clinics = clinics.replace(-1, np.nan)
    clinics["dbp"] = clinics["dbp"].map(lambda x: np.nan if x in [0, 1] else x)
    clinics["sbp"] = clinics["sbp"].map(lambda x: np.nan if x in [0, 1] else x)
    clinics["sbp"] = clinics["sbp"].map(lambda x: np.nan if x > 1000 else x)
    clinics["v"] = clinics["v"].replace({"E": np.nan, "T": np.nan})
    # clinics.fillna(0, inplace=True)

    # change dtypes
    clinics["platet"] = clinics["platet"].astype(float)
    clinics["v"] = clinics["v"].astype(float)
    clinics["sex"] = clinics["sex"].astype(float)
    clinics["age"] = clinics["age"].astype(float)
    clinics["ivh"] = clinics["ivh"].astype(float)

    
    # merge ids, nifti_path, label to clinics
    clinics["id"] = df["id"].astype(str)
    clinics["image"] = df['id'].apply(lambda id: os.path.join(image_path, f"{id}.nii.gz"))
    clinics["image"] = clinics["image"].astype(str)
    clinics["mask"] = df['id'].apply(lambda id: os.path.join(mask_path, f"{id}.nii.gz"))
    clinics["mask"] = clinics["mask"].astype(str)
    clinics["label"] = df["label"].astype(int)
    
    # remove rows s.t. nifti_path does not exist
    clinics = clinics[clinics['image'].apply(os.path.exists)]

    return clinics

if __name__ == "__main__":
    # 돌리기 전에 segment mask 파일들이 환자번호로만 존재하게 만들기 -- deepbleed/voxel_vol_EDA.ipynb 파일에 만들어둠
    df_path = "/ssd_data1/syjeong/hematoma/data/EHR/snuh_ehr.csv"
    image_path = "/ssd_data1/syjeong/hematoma/data/stripped/internal/pre"
    mask_path = "/ssd_data1/syjeong/hematoma/data/segMask/internal/pre"
    data = snuh_prep(df_path, image_path, mask_path)
    data = data.reset_index(drop=True)
    data, imputer = impute_data(
        data,
        mode="train",
        impute_methods="knn",
        save_path="/ssd_data1/syjeong/hematoma/code/drfuse_HE/experiments/sample",
    )
    data, scaler = scale_data(data, mode="train",save_path="/ssd_data1/syjeong/hematoma/code/drfuse_HE/experiments/sample")
    data = dict_list(data)
    print(data[0])
    df_path = "/ssd_data1/syjeong/hematoma/data/EHR/ehr_brm.csv"
    image_path = "/ssd_data1/syjeong/hematoma/data/stripped/external/pre"
    mask_path = "/ssd_data1/syjeong/hematoma/data/segMask/external/pre"
    data = brm_prep(df_path, image_path, mask_path) 
    # data = snuh_prep(df_path, image_path, mask_path)
    data = data.reset_index(drop=True)
    data = impute_data(
        data,
        mode="test",
        impute_methods="knn",
        imputer = imputer,
        # save_path="/ssd_data1/syjeong/hematoma/code/drfuse_HE/experiments/sample/imputer.joblib",
    )
    data = scale_data(data, mode="test",
    scaler=scaler,
    # save_path="/ssd_data1/syjeong/hematoma/code/drfuse_HE/experiments/sample/scaler.joblib",
    )
    data = dict_list(data)
    print(data[0])