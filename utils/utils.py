import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def create_feature_space(df, features_list):
    if "esm2" in features_list:
        data = pd.read_parquet(f"data/precomputed_embeddings/gdpa1_esm2_mean_no_sep_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")

    if "esm2_cls" in features_list:
        data = pd.read_parquet(f"data/precomputed_embeddings/gdpa1_esm2_cls_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")
    
    if "lbster" in features_list:
        data = pd.read_parquet(f"data/precomputed_embeddings/gdpa1_lbster_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")
    
    if "ablang2" in features_list:
        data = pd.read_parquet(f"data/precomputed_embeddings/gdpa1_ablang2_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")

    if "moe" in features_list:
        data = pd.read_csv(f"data/provided_features/MOE_properties_gdpa1.csv")
        # update the col names to prefix the property cols with moe
        data.columns = ["antibody_id", "antibody_name"] + ["moe_" + x for x in data.columns[2:]]
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")
    
    if 'tap' in features_list:
        data = pd.read_csv(f"data/provided_features/TAP_gdpa1.csv")
        data.columns = ["antibody_name"] + ["tap_" + x for x in data.columns[1:]]
        df = df.merge(data, on=["antibody_name"], how="left")

    return df

def create_feature_space_test_set(df, features_list):
    if "esm2" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/test_set/test_set_esm2_mean_no_sep_embeddings.parquet")
        df = df.merge(data, on=["antibody_name"], how="left")

    if "esm2_cls" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/test_set/test_set_esm2_cls_embeddings.parquet")
        df = df.merge(data, on=["antibody_name"], how="left")

    if "lbster" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/test_set/test_set_lbster_embeddings.parquet")
        df = df.merge(data, on=["antibody_name"], how="left")
    
    if "ablang2" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/test_set/test_set_ablang2_embeddings.parquet")
        df = df.merge(data, on=["antibody_name"], how="left")

    if "moe" in features_list:
        data = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/MOE_properties_test_set.csv")
        # update the col names to prefix the property cols with moe
        data.columns = ["antibody_name"] + ["moe_" + x for x in data.columns[1:]]
        df = df.merge(data, on=["antibody_name"], how="left")
    
    if 'tap' in features_list:
        data = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/TAP_test_set.csv")
        data.columns = ["antibody_name"] + ["tap_" + x for x in data.columns[1:]]
        df = df.merge(data, on=["antibody_name"], how="left")

    if "seq" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/test_set_seq_features.parquet")
        df = df.merge(data, on=["antibody_name"], how="left")

    return df





def create_feature_space_full_seqs(df, features_list):
    if "esm2" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/full_length/gdpa1_full_esm2_mean_no_sep_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")

        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/full_length/gdpa1_full_esm2_cls_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")
    
    if "ablang2" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/full_length/gdpa1_full_ablang2_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")

    if "igbert" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/full_length/gdpa1_full_igbert_mean_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")

    if "antiberta2" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/full_length/gdpa1_full_antiberta2_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")
    
    if "antiberta2_cssp" in features_list:
        data = pd.read_parquet("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/precomputed_embeddings/full_length/gdpa1_full_antiberta2_cssp_embeddings.parquet")
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")

    if "moe" in features_list:
        data = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/MOE_properties.csv")
        # update the col names to prefix the property cols with moe
        data.columns = ["antibody_id", "antibody_name"] + ["moe_" + x for x in data.columns[2:]]
        df = df.merge(data, on=["antibody_id", "antibody_name"], how="left")

    return df