# Databricks notebook source
!pip install -r requirements_embeddings.txt

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Literal, Any, Optional
import torch
import gc
from tqdm import tqdm
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ### read in sequences for embeddings

# COMMAND ----------

df = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/GDPa1_v1.2_20250814.csv")
print(df.shape)
df.head(2)

df = df.rename(columns=({"SEC %Monomer": "SEC_perc_Monomer",
                         "AC-SINS_pH6.0": "AC_SINS_pH6.0",
                         "AC-SINS_pH7.4": "AC_SINS_pH7.4",
                         "vh_protein_sequence": "vh",
                         "vl_protein_sequence": "vl"}))

# df.isna().sum()
lower_cols = [x.lower() for x in df.columns]
# print(lower_cols)
df.columns = lower_cols

df.head(2)


# COMMAND ----------

display(df.groupby(by=["hc_subtype"],as_index=False).agg({"antibody_id": "count"}))
display(df.groupby(by=["lc_subtype"],as_index=False).agg({"antibody_id": "count"}))
print(df["vh"].nunique())
print(df["vl"].nunique())

# COMMAND ----------

# reading in data from gdb_mart
dg = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/biodeva-modeling/nbs/GDB_mart_data_EDA/data/mature_chain_sequences_for_anarci.csv")
print(dg.shape)
dg.head(2)

dg = dg.loc[dg["Name"].isin(df["tpp"])]
print(dg.shape)
dg.head(2)

dg["combined_sequences"] = dg["Heavy_Chain"] + "<SEP>" + dg["Light_Chain"]
sequences = dg["combined_sequences"].to_list()
print(len(sequences))


# COMMAND ----------

# keep only tpps in the biodeva dataset
df = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/biodeva-modeling/nbs/compare_PLMs/data/biodeva_all_projects_combined.csv")

print(df.shape)

df = df.drop_duplicates(subset=["tpp"]).reset_index(drop=True)
print(df.shape)
df.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC # ESM2 embeddings

# COMMAND ----------

df["fv"] = df["vh"] + "<SEP>" + df["vl"]
sequences = df["fv"].to_list()
print(len(sequences))

# COMMAND ----------

from transformers import EsmModel, EsmTokenizer
class ESM2FeatureExtractor:
    """ESM2-based feature extractor with multiple pooling strategies."""
    
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D"):
        """
        Initialize ESM2 feature extractor.
        
        Args:
            model_name: ESM2 model name from HuggingFace
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = EsmTokenizer.from_pretrained(model_name)
            self.model = EsmModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            # Freeze parameters to save memory
            for param in self.model.parameters():
                param.requires_grad = False
                
            print(f"Model loaded successfully. Embedding dimension: {self.model.config.hidden_size}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ESM2 model '{model_name}': {str(e)}")
    
    def extract_embeddings(self, 
                          sequences: List[str], 
                          batch_size: int = 8,
                          pooling_strategy: Literal["cls", "mean", "mean_no_special", "mean_no_sep", "cls_mean"] = "cls") -> np.ndarray:
        """
        Extract ESM2 embeddings for protein sequences with different pooling strategies.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
            pooling_strategy: How to pool sequence embeddings:
                - "cls": Use only the <cls> token (recommended for classification)
                - "mean": Mean pool over all tokens (including separator)
                - "mean_no_special": Mean pool excluding <cls> and <eos> tokens only
                - "mean_no_sep": Mean pool excluding <cls>, <eos>, and <sep> tokens
                - "cls_mean": Concatenate <cls> token with mean pooled embeddings
            
        Returns:
            numpy array of embeddings
        """
        if not sequences:
            raise ValueError("No sequences provided")
            
        embeddings = []
        
        # Auto-adjust batch size for GPU memory
        if self.device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 8 and batch_size > 4:
                batch_size = 4
                print(f"Reduced batch size to {batch_size} for GPU with {gpu_memory_gb:.1f}GB memory")
        
        print(f"Using pooling strategy: {pooling_strategy}")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
                batch_sequences = sequences[i:i+batch_size]
                
                try:
                    # Tokenize sequences
                    inputs = self.tokenizer(
                        batch_sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024
                    ).to(self.device)
                    
                    # Get embeddings
                    outputs = self.model(**inputs)
                    batch_embeddings = self._apply_pooling_strategy(
                        outputs, inputs, pooling_strategy
                    ).cpu().numpy()
                    
                    embeddings.extend(batch_embeddings)
                    
                except torch.cuda.OutOfMemoryError:
                    # Handle OOM by reducing batch size and retrying
                    torch.cuda.empty_cache()
                    if batch_size > 1:
                        print(f"OOM error, reducing batch size from {batch_size} to {batch_size//2}")
                        batch_size = batch_size // 2
                        # Reprocess this batch with smaller size
                        for k in range(0, len(batch_sequences), batch_size):
                            mini_batch = batch_sequences[k:k+batch_size]
                            inputs = self.tokenizer(
                                mini_batch,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=1024
                            ).to(self.device)
                            
                            outputs = self.model(**inputs)
                            mini_batch_embeddings = self._apply_pooling_strategy(
                                outputs, inputs, pooling_strategy
                            ).cpu().numpy()
                            embeddings.extend(mini_batch_embeddings)
                    else:
                        raise RuntimeError("Out of memory even with batch size 1")
                
                except Exception as e:
                    raise RuntimeError(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                
                # Clean up GPU memory after each batch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        embeddings_array = np.array(embeddings)
        print(f"Extracted embeddings shape: {embeddings_array.shape}")
        return embeddings_array
    
    def _apply_pooling_strategy(self, outputs, inputs, strategy: str) -> torch.Tensor:
        """Apply the specified pooling strategy to get sequence representations."""
        embeddings_tensor = outputs.last_hidden_state  # Shape: [batch, seq_len, hidden_dim]
        attention_mask = inputs['attention_mask']
        
        if strategy == "cls":
            # Use only the <cls> token (first position)
            return embeddings_tensor[:, 0, :]
        
        elif strategy == "mean":
            # Simple mean pooling over all non-padding tokens
            # Attention mask handles padding automatically
            masked_embeddings = embeddings_tensor * attention_mask.unsqueeze(-1)
            sequence_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            sequence_lengths = torch.clamp(sequence_lengths, min=1)
            return masked_embeddings.sum(dim=1) / sequence_lengths
        
        elif strategy == "mean_no_special":
            # Mean pooling excluding <cls> and <eos> tokens
            sequence_mask = attention_mask.clone().float()
            
            # Exclude <cls> token (first position)
            sequence_mask[:, 0] = 0
            
            # Exclude <eos> token (last non-padding token)
            for j, seq_len in enumerate(attention_mask.sum(dim=1)):
                if seq_len > 1:
                    sequence_mask[j, seq_len-1] = 0
            
            masked_embeddings = embeddings_tensor * sequence_mask.unsqueeze(-1)
            sequence_lengths = sequence_mask.sum(dim=1, keepdim=True)
            sequence_lengths = torch.clamp(sequence_lengths, min=1)
            return masked_embeddings.sum(dim=1) / sequence_lengths
        
        elif strategy == "mean_no_sep":
            # Mean pooling excluding <cls>, <eos>, and <sep> tokens
            input_ids = inputs['input_ids']
            sequence_mask = attention_mask.clone().float()
            
            # Exclude <cls> token
            sequence_mask[:, 0] = 0
            
            # Exclude <eos> token
            for j, seq_len in enumerate(attention_mask.sum(dim=1)):
                if seq_len > 1:
                    sequence_mask[j, seq_len-1] = 0
            
            # Exclude <SEP> tokens
            sep_token_id = self.tokenizer.convert_tokens_to_ids('<sep>')
            if sep_token_id is not None:
                sep_positions = (input_ids == sep_token_id)
                sequence_mask = sequence_mask * (~sep_positions).float()
            
            masked_embeddings = embeddings_tensor * sequence_mask.unsqueeze(-1)
            sequence_lengths = sequence_mask.sum(dim=1, keepdim=True)
            sequence_lengths = torch.clamp(sequence_lengths, min=1)
            return masked_embeddings.sum(dim=1) / sequence_lengths
        
        elif strategy == "cls_mean":
            # Concatenate <cls> token with mean pooled embeddings
            cls_embeddings = embeddings_tensor[:, 0, :]
            
            # Mean pool (simple version)
            masked_embeddings = embeddings_tensor * attention_mask.unsqueeze(-1)
            sequence_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            sequence_lengths = torch.clamp(sequence_lengths, min=1)
            mean_embeddings = masked_embeddings.sum(dim=1) / sequence_lengths
            
            return torch.cat([cls_embeddings, mean_embeddings], dim=1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
    
    def get_embedding_dim(self, pooling_strategy: str = "cls") -> int:
        """Get the dimensionality of the embeddings for a given pooling strategy."""
        base_dim = self.model.config.hidden_size
        if pooling_strategy == "cls_mean":
            return base_dim * 2  # Concatenated <cls> + mean
        else:
            return base_dim
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction


# Utility function to compare pooling strategies
def compare_pooling_strategies(sequences: List[str], 
                             strategies: List[str] = ["cls", "mean", "mean_no_special", "mean_no_sep", "cls_mean"]) -> dict:
    """
    Compare different pooling strategies on the same sequences.
    
    Args:
        sequences: List of protein sequences to test
        strategies: List of pooling strategies to compare
        
    Returns:
        Dictionary with strategy names as keys and embeddings as values
    """
    extractor = ESM2FeatureExtractor()
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        embeddings = extractor.extract_embeddings(sequences, pooling_strategy=strategy)
        results[strategy] = embeddings
        print(f"Shape: {embeddings.shape}")
    
    return results

# COMMAND ----------

# --- Your sequences and function call ---

# Create an instance of the extractor
extractor = ESM2FeatureExtractor()

# Get embeddings with 'mean_no_sep' pooling
# mean_no_sep_embeddings = extractor.extract_embeddings(
#     sequences=sequences, 
#     pooling_strategy="mean_no_sep"
# )
cls_embeddings = extractor.extract_embeddings(
    sequences=sequences, 
    pooling_strategy="cls"
)

# Print the resulting embeddings and their shape
print("\nShape of the final embeddings array:", cls_embeddings.shape)

# Clean up to release GPU memory (good practice)
extractor.cleanup()

# COMMAND ----------

len(mean_no_sep_embeddings)
len(cls_embeddings)

# COMMAND ----------

len(cls_embeddings[0])

# COMMAND ----------

# saving the combined embeddings
# embeddings = [emb for emb in mean_no_sep_embeddings]
embeddings = [emb for emb in cls_embeddings]

# Stack the arrays to create a single array
embeddings_array = np.vstack(embeddings)
print(embeddings_array.shape)

emb_df = pd.DataFrame(embeddings_array)
print(emb_df.shape)
cols = ["ems2_cls_" + str(x) for x in emb_df.columns]
# print(cols)
emb_df.columns = cols

de = pd.concat([df[["antibody_id", "antibody_name"]], emb_df], axis=1)
print(de.shape)
de.head(2)


# COMMAND ----------

import pyarrow as pa
import pyarrow.parquet as pq

# de.to_parquet('gdpa1_esm2_mean_no_sep_embeddings.parquet', engine='pyarrow', index=False)
de.to_parquet('gdpa1_esm2_cls_embeddings.parquet', engine='pyarrow', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # AntiBERTa2

# COMMAND ----------

import torch
import numpy as np
from typing import List, Literal
from tqdm import tqdm
import gc


class AntiBERTaFeatureExtractor:
    """AntiBERTa-based feature extractor for antibody sequences."""
    
    def __init__(self, model_name: str = "alchemab/antiberta2"):
        """
        Initialize AntiBERTa feature extractor.
        
        Args:
            model_name: AntiBERTa model name from HuggingFace
                       Options: 
                       - "alchemab/antiberta2" (standard AntiBERTa2)
                       - "alchemab/antiberta2-cssp" (with structural information)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            from transformers import RoFormerTokenizer, RoFormerModel
            
            self.tokenizer = RoFormerTokenizer.from_pretrained(model_name)
            self.model = RoFormerModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            # Freeze parameters to save memory
            for param in self.model.parameters():
                param.requires_grad = False
                
            print(f"AntiBERTa model loaded successfully: {model_name}")
            print(f"Embedding dimension: {self.model.config.hidden_size}")
            
            # Check for non-commercial use license
            print("⚠️  Note: AntiBERTa2 models are available for NON-COMMERCIAL USE ONLY")
            print("For commercial use, contact info@alchemab.com")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load AntiBERTa model '{model_name}': {str(e)}")
    
    def extract_embeddings(self, 
                          sequences: List[str], 
                          batch_size: int = 8,
                          pooling_strategy: Literal["mean", "cls"] = "mean") -> np.ndarray:
        """
        Extract AntiBERTa embeddings for antibody sequences.
        
        Args:
            sequences: List of antibody sequences (heavy chain, light chain, or CDR sequences)
            batch_size: Batch size for processing
            pooling_strategy: How to pool sequence embeddings:
                - "mean": Mean pool over all tokens (recommended for antibodies)
                - "cls": Use only the [CLS] token
            
        Returns:
            numpy array of embeddings
        """
        if not sequences:
            raise ValueError("No sequences provided")
            
        embeddings = []
        
        # Auto-adjust batch size for GPU memory
        if self.device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 8 and batch_size > 4:
                batch_size = 4
                print(f"Reduced batch size to {batch_size} for GPU with {gpu_memory_gb:.1f}GB memory")
        
        print(f"Using pooling strategy: {pooling_strategy} (recommended: mean)")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting AntiBERTa embeddings"):
                batch_sequences = sequences[i:i+batch_size]
                
                try:
                    # Tokenize sequences
                    inputs = self.tokenizer(
                        batch_sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512  # AntiBERTa typically uses shorter sequences
                    ).to(self.device)
                    
                    # Get embeddings
                    outputs = self.model(**inputs)
                    batch_embeddings = self._apply_pooling_strategy(
                        outputs, inputs, pooling_strategy
                    ).cpu().numpy()
                    
                    embeddings.extend(batch_embeddings)
                    
                except torch.cuda.OutOfMemoryError:
                    # Handle OOM by reducing batch size and retrying
                    torch.cuda.empty_cache()
                    if batch_size > 1:
                        print(f"OOM error, reducing batch size from {batch_size} to {batch_size//2}")
                        batch_size = batch_size // 2
                        # Reprocess this batch with smaller size
                        for k in range(0, len(batch_sequences), batch_size):
                            mini_batch = batch_sequences[k:k+batch_size]
                            inputs = self.tokenizer(
                                mini_batch,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512
                            ).to(self.device)
                            
                            outputs = self.model(**inputs)
                            mini_batch_embeddings = self._apply_pooling_strategy(
                                outputs, inputs, pooling_strategy
                            ).cpu().numpy()
                            embeddings.extend(mini_batch_embeddings)
                    else:
                        raise RuntimeError("Out of memory even with batch size 1")
                
                except Exception as e:
                    raise RuntimeError(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                
                # Clean up GPU memory after each batch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        embeddings_array = np.array(embeddings)
        print(f"Extracted AntiBERTa embeddings shape: {embeddings_array.shape}")
        return embeddings_array
    
    def _apply_pooling_strategy(self, outputs, inputs, strategy: str) -> torch.Tensor:
        """Apply the specified pooling strategy to get sequence representations."""
        embeddings_tensor = outputs.last_hidden_state  # Shape: [batch, seq_len, hidden_dim]
        attention_mask = inputs['attention_mask']
        
        if strategy == "mean":
            # Mean pooling over all non-padding tokens (recommended for antibodies)
            # This captures the full sequence information which is important for antibody function
            masked_embeddings = embeddings_tensor * attention_mask.unsqueeze(-1)
            sequence_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            sequence_lengths = torch.clamp(sequence_lengths, min=1)
            return masked_embeddings.sum(dim=1) / sequence_lengths
        
        elif strategy == "cls":
            # Use only the [CLS] token (first position)
            return embeddings_tensor[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}. Use 'mean' (recommended) or 'cls'.")
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.model.config.hidden_size
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction


# Utility function for antibody-specific processing
def extract_antibody_embeddings(sequences: List[str], 
                               model_name: str = "alchemab/antiberta2",
                               batch_size: int = 8,
                               pooling_strategy: str = "mean") -> np.ndarray:
    """
    Convenience function to extract antibody embeddings using AntiBERTa.
    
    Args:
        sequences: List of antibody sequences (heavy chain, light chain, or CDR sequences)
        model_name: AntiBERTa model to use
        batch_size: Batch size for processing
        pooling_strategy: Pooling strategy to use
        
    Returns:
        numpy array of embeddings using the specified pooling strategy
    """
    extractor = AntiBERTaFeatureExtractor(model_name=model_name)
    embeddings = extractor.extract_embeddings(
        sequences, 
        batch_size=batch_size, 
        pooling_strategy=pooling_strategy
    )
    extractor.cleanup()
    return embeddings


def extract_separate_chain_embeddings(heavy_chains: List[str], 
                                     light_chains: List[str],
                                     model_name: str = "alchemab/antiberta2",
                                     batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings for heavy and light chains separately (recommended for AntiBERTa).
    
    Args:
        heavy_chains: List of heavy chain sequences
        light_chains: List of light chain sequences
        model_name: AntiBERTa model to use
        batch_size: Batch size for processing
        
    Returns:
        tuple of (heavy_embeddings, light_embeddings)
    """
    if len(heavy_chains) != len(light_chains):
        print("Warning: Number of heavy and light chains differ. Processing independently.")
    
    print("Extracting heavy chain embeddings...")
    heavy_embeddings = extract_antibody_embeddings(heavy_chains, model_name, batch_size)
    
    print("Extracting light chain embeddings...")
    light_embeddings = extract_antibody_embeddings(light_chains, model_name, batch_size)
    
    return heavy_embeddings, light_embeddings


def extract_combined_embeddings(heavy_chains: List[str], 
                               light_chains: List[str],
                               model_name: str = "alchemab/antiberta2",
                               batch_size: int = 8,
                               combination_method: str = "concatenate") -> np.ndarray:
    """
    Extract and combine heavy and light chain embeddings.
    
    Args:
        heavy_chains: List of heavy chain sequences
        light_chains: List of light chain sequences
        model_name: AntiBERTa model to use
        batch_size: Batch size for processing
        combination_method: How to combine embeddings ("concatenate" or "add")
        
    Returns:
        Combined embeddings array
    """
    heavy_embeddings, light_embeddings = extract_separate_chain_embeddings(
        heavy_chains, light_chains, model_name, batch_size
    )
    
    # Ensure same number of embeddings
    min_len = min(len(heavy_embeddings), len(light_embeddings))
    heavy_embeddings = heavy_embeddings[:min_len]
    light_embeddings = light_embeddings[:min_len]
    
    if combination_method == "concatenate":
        combined_embeddings = np.concatenate([heavy_embeddings, light_embeddings], axis=1)
        print(f"Concatenated embeddings shape: {combined_embeddings.shape}")
    elif combination_method == "add":
        combined_embeddings = heavy_embeddings + light_embeddings
        print(f"Added embeddings shape: {combined_embeddings.shape}")
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")
    
    return combined_embeddings


# COMMAND ----------

df["fv"] = df["vh"] + "[SEP]" + df["vl"]
sequences = df["fv"].to_list()

# COMMAND ----------

# MAGIC %md
# MAGIC #### paired usage

# COMMAND ----------

print("\n=== Manual AntiBERTa2 Paired Chains Usage ===")
extractor = AntiBERTaFeatureExtractor(model_name="alchemab/antiberta2")
antibert_embeddings = extractor.extract_embeddings(sequences, pooling_strategy="mean")
print(f"Manual paired embeddings shape: {antibert_embeddings.shape}")
extractor.cleanup()

# COMMAND ----------

# saving the combined embeddings
embeddings = [emb for emb in antibert_embeddings]

# Stack the arrays to create a single array
embeddings_array = np.vstack(embeddings)
print(embeddings_array.shape)

emb_df = pd.DataFrame(embeddings_array)
print(emb_df.shape)
cols = ["antiberta2_" + str(x) for x in emb_df.columns]
# print(cols)
emb_df.columns = cols

de = pd.concat([df[["antibody_id", "antibody_name"]], emb_df], axis=1)
print(de.shape)
de.head(2)


# COMMAND ----------

import pyarrow as pa
import pyarrow.parquet as pq

de.to_parquet('gdpa1_antiberta2_embeddings.parquet', engine='pyarrow', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### AntiBERTa2 structural info

# COMMAND ----------

# Get structural info Antiberta2-CSSP embeddings
print("\n=== Using AntiBERTa2-CSSP (with structural information) ===")
try:
    cssp_embeddings = extract_antibody_embeddings(sequences, model_name="alchemab/antiberta2-cssp")
    print(f"CSSP embeddings shape: {cssp_embeddings.shape}")
except Exception as e:
    print(f"CSSP model error: {e}")

# COMMAND ----------

# saving the combined embeddings
embeddings = [emb for emb in cssp_embeddings]

# Stack the arrays to create a single array
embeddings_array = np.vstack(embeddings)
print(embeddings_array.shape)

emb_df = pd.DataFrame(embeddings_array)
print(emb_df.shape)
cols = ["antiberta2_cssp_" + str(x) for x in emb_df.columns]
emb_df.columns = cols

de = pd.concat([df[["antibody_id", "antibody_name"]], emb_df], axis=1)
print(de.shape)
de.head(2)

de.to_parquet('gdpa1_antiberta2_cssp_embeddings.parquet', engine='pyarrow', index=False)


# COMMAND ----------

de.to_parquet('gdpa1_antiberta2_cssp_embeddings.parquet', engine='pyarrow', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # IgBERT

# COMMAND ----------

import torch
import numpy as np
from typing import List, Literal
from tqdm import tqdm
import gc

class IgBertFeatureExtractor:
    """IgBert-based feature extractor for antibody sequences."""
    
    def __init__(self, model_name: str = "Exscientia/IgBert", paired: bool = True):
        """
        Initialize IgBert feature extractor.
        
        Args:
            model_name: IgBert model name from HuggingFace
                       Options: "Exscientia/IgBert" (paired version, recommended)
                               "Exscientia/IgBert_unpaired" (unpaired version)
            paired: Whether to use the paired version that can handle heavy+light chain sequences
        """
        # Select the appropriate model based on paired parameter
        if paired and "unpaired" not in model_name:
            self.model_name = model_name  # Use paired version
        elif not paired and "unpaired" not in model_name:
            self.model_name = "Exscientia/IgBert_unpaired"  # Use unpaired version with correct name
        else:
            self.model_name = model_name  # Use as specified
            
        self.paired = paired
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.model = BertModel.from_pretrained(self.model_name, add_pooling_layer=False).to(self.device)
            self.model.eval()
            
            # Freeze parameters to save memory
            for param in self.model.parameters():
                param.requires_grad = False
                
            print(f"IgBert model loaded successfully: {self.model_name}")
            print(f"Embedding dimension: {self.model.config.hidden_size}")
            print(f"Paired mode: {self.paired}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load IgBert model '{self.model_name}': {str(e)}")
    
    def extract_embeddings(self, 
                          sequences: List[str], 
                          batch_size: int = 8,
                          pooling_strategy: Literal["mean", "cls"] = "mean") -> np.ndarray:
        """
        Extract IgBert embeddings for antibody sequences.
        
        Args:
            sequences: List of antibody sequences. For paired mode, use format "HEAVY_CHAIN<sep>LIGHT_CHAIN"
                      For unpaired mode, use individual heavy or light chain sequences
            batch_size: Batch size for processing
            pooling_strategy: How to pool sequence embeddings:
                - "mean": Mean pool over all tokens (recommended for antibodies)
                - "cls": Use only the [CLS] token
            
        Returns:
            numpy array of embeddings
        """
        if not sequences:
            raise ValueError("No sequences provided")
            
        embeddings = []
        
        # Auto-adjust batch size for GPU memory
        if self.device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 8 and batch_size > 4:
                batch_size = 4
                print(f"Reduced batch size to {batch_size} for GPU with {gpu_memory_gb:.1f}GB memory")
        
        print(f"Using pooling strategy: {pooling_strategy} (recommended: mean)")
        if self.paired:
            print("Note: For paired mode, format sequences as 'HEAVY_CHAIN<sep>LIGHT_CHAIN'")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting IgBert embeddings"):
                batch_sequences = sequences[i:i+batch_size]
                
                try:
                    # Tokenize sequences
                    inputs = self.tokenizer(
                        batch_sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024  # IgBert can handle longer sequences than typical BERT models
                    ).to(self.device)
                    
                    # Get embeddings
                    outputs = self.model(**inputs)
                    batch_embeddings = self._apply_pooling_strategy(
                        outputs, inputs, pooling_strategy
                    ).cpu().numpy()
                    
                    embeddings.extend(batch_embeddings)
                    
                except torch.cuda.OutOfMemoryError:
                    # Handle OOM by reducing batch size and retrying
                    torch.cuda.empty_cache()
                    if batch_size > 1:
                        print(f"OOM error, reducing batch size from {batch_size} to {batch_size//2}")
                        batch_size = batch_size // 2
                        # Reprocess this batch with smaller size
                        for k in range(0, len(batch_sequences), batch_size):
                            mini_batch = batch_sequences[k:k+batch_size]
                            inputs = self.tokenizer(
                                mini_batch,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=1024
                            ).to(self.device)
                            
                            outputs = self.model(**inputs)
                            mini_batch_embeddings = self._apply_pooling_strategy(
                                outputs, inputs, pooling_strategy
                            ).cpu().numpy()
                            embeddings.extend(mini_batch_embeddings)
                    else:
                        raise RuntimeError("Out of memory even with batch size 1")
                
                except Exception as e:
                    raise RuntimeError(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                
                # Clean up GPU memory after each batch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        embeddings_array = np.array(embeddings)
        print(f"Extracted IgBert embeddings shape: {embeddings_array.shape}")
        return embeddings_array
    
    def _apply_pooling_strategy(self, outputs, inputs, strategy: str) -> torch.Tensor:
        """Apply the specified pooling strategy to get sequence representations."""
        embeddings_tensor = outputs.last_hidden_state  # Shape: [batch, seq_len, hidden_dim]
        attention_mask = inputs['attention_mask']
        
        if strategy == "mean":
            # Mean pooling over all non-padding tokens (recommended for antibodies)
            # This captures the full sequence information which is important for antibody function
            masked_embeddings = embeddings_tensor * attention_mask.unsqueeze(-1)
            sequence_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            sequence_lengths = torch.clamp(sequence_lengths, min=1)
            return masked_embeddings.sum(dim=1) / sequence_lengths
        
        elif strategy == "cls":
            # Use only the [CLS] token (first position)
            return embeddings_tensor[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}. Use 'mean' (recommended) or 'cls'.")
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.model.config.hidden_size
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction


# Utility functions for different use cases
def extract_paired_antibody_embeddings(heavy_chains: List[str], 
                                      light_chains: List[str],
                                      model_name: str = "Exscientia/IgBert",
                                      batch_size: int = 8,
                                      separator: str = "<sep>") -> np.ndarray:
    """
    Extract embeddings for paired heavy and light chain sequences using IgBert.
    
    Args:
        heavy_chains: List of heavy chain sequences
        light_chains: List of light chain sequences
        model_name: IgBert model to use
        batch_size: Batch size for processing
        separator: Separator token between heavy and light chains
        
    Returns:
        numpy array of paired embeddings
    """
    if len(heavy_chains) != len(light_chains):
        raise ValueError("Number of heavy and light chains must match")
    
    # Combine heavy and light chains with separator
    paired_sequences = [f"{h}{separator}{l}" for h, l in zip(heavy_chains, light_chains)]
    
    extractor = IgBertFeatureExtractor(model_name=model_name, paired=True)
    embeddings = extractor.extract_embeddings(
        paired_sequences, 
        batch_size=batch_size, 
        pooling_strategy="mean"
    )
    extractor.cleanup()
    return embeddings

def extract_unpaired_antibody_embeddings(sequences: List[str],
                                        model_name: str = "Exscientia/IgBert_unpaired", 
                                        batch_size: int = 8) -> np.ndarray:
    """
    Extract embeddings for individual antibody sequences (heavy or light chains) using IgBert.
    
    Args:
        sequences: List of individual antibody sequences
        model_name: IgBert unpaired model to use
        batch_size: Batch size for processing
        
    Returns:
        numpy array of embeddings
    """
    extractor = IgBertFeatureExtractor(model_name=model_name, paired=False)
    embeddings = extractor.extract_embeddings(
        sequences, 
        batch_size=batch_size, 
        pooling_strategy="mean"
    )
    extractor.cleanup()
    return embeddings




# COMMAND ----------

heavy_chains = df["vh"].to_list()
light_chains = df["vl"].to_list()

# COMMAND ----------

igbert_embeddings = extract_paired_antibody_embeddings(heavy_chains, light_chains)
print(f"Paired IgBERT embeddings shape: {igbert_embeddings.shape}")

# COMMAND ----------

# saving the combined embeddings
embeddings = [emb for emb in igbert_embeddings]

# Stack the arrays to create a single array
embeddings_array = np.vstack(embeddings)
print(embeddings_array.shape)

emb_df = pd.DataFrame(embeddings_array)
print(emb_df.shape)
cols = ["igbert_" + str(x) for x in emb_df.columns]
emb_df.columns = cols

de = pd.concat([df[["antibody_id", "antibody_name"]], emb_df], axis=1)
print(de.shape)
de.head(2)

# de.to_parquet('gdpa1_antiberta2_cssp_embeddings.parquet', engine='pyarrow', index=False)


# COMMAND ----------

de.to_parquet('gdpa1_igbert_mean_embeddings.parquet', engine='pyarrow', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # LBSTER embeddings

# COMMAND ----------

df = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/heldout-set-sequences.csv")
print(df.shape)
df.head(2)

df = df.rename(columns=({"vh_protein_sequence": "vh",
                         "vl_protein_sequence": "vl"}))

print(df.isna().sum())
# df.head(2)

# COMMAND ----------

df = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/biodeva-modeling/nbs/non_antibody_titer_classification/data/filtered_data_with_seqs_non_antibody_proteins.csv")
print(df.shape)
print(df["ID"].nunique())
print(df["complex_seq"].nunique())
df.head(2)

dt = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/biodeva-modeling/nbs/non_antibody_titer_classification/data/new_test_seqs_non_antibody_proteins.csv")
print(dt.shape)
dt.head(2)

# COMMAND ----------

# df["fv"] = df["vh"] + "[SEP]" + df["vl"]
# sequences = df["fv"].to_list()

sequences = dt['complex_seq'].to_list()
print(len(sequences))

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo apt-get update && sudo apt-get install -y libaio-dev

# COMMAND ----------

# MAGIC %pip install --upgrade transformers accelerate
# MAGIC %pip uninstall -y deepspeed
# MAGIC %pip install deepspeed --no-cache-dir

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys
import transformers

if not hasattr(transformers, 'modeling_layers'):
    import transformers.modeling_utils as modeling_utils
    sys.modules['transformers.modeling_layers'] = modeling_utils

print("Setup complete!")

# COMMAND ----------

from transformers import Trainer, TrainingArguments


# COMMAND ----------

# MAGIC %md
# MAGIC #### paired lbster embeddings

# COMMAND ----------

from lobster.model import LobsterPMLM

# Determine the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the LobsterPMLM model
lobster = LobsterPMLM("asalam91/lobster_24M").to(device)
lobster.eval()

# Set a reasonable batch size based on your machine's memory
batch_size = 128  # Adjust this number based on cluster memory
all_pooled_embeddings = []

# Process the data in batches
for i in range(0, len(sequences), batch_size):
    batch_sequences = sequences[i:i + batch_size]
    
    # Process the batch
    with torch.no_grad(): # Disable gradient calculation for faster inference and less memory
        batch_mlm_representations = lobster.sequences_to_latents(batch_sequences)[-1]

    # Pool the representations for the current batch
    batch_pooled_embeddings = torch.mean(batch_mlm_representations, dim=1)
    
    # Append the results of the current batch to the main list
    all_pooled_embeddings.append(batch_pooled_embeddings)

# Concatenate all the batch results into a single tensor
paired_embeddings_tensor = torch.cat(all_pooled_embeddings, dim=0)

# Convert the tensor to a NumPy array and save it
paired_embeddings_np = paired_embeddings_tensor.cpu().detach().numpy()

# print(f"Combined embeddings successfully saved to paired_lbster_embeddings_biodeva.npy")
print(f"Shape of the final NumPy array: {paired_embeddings_np.shape}")

# COMMAND ----------

dt.head(2)

# COMMAND ----------

# saving the combined embeddings
# embeddings = [emb for emb in igbert_embeddings]

# # Stack the arrays to create a single array
# embeddings_array = np.vstack(embeddings)
# print(embeddings_array.shape)

emb_df = pd.DataFrame(paired_embeddings_np)
print(emb_df.shape)
cols = ["lbster_" + str(x) for x in emb_df.columns]
emb_df.columns = cols

de = pd.concat([dt[["ID"]], emb_df], axis=1)
print(de.shape)
de.head(2)

# COMMAND ----------

# de.to_parquet('test_set_lbster_embeddings.parquet', engine='pyarrow', index=False)
de.to_parquet('/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/biodeva-modeling/nbs/non_antibody_titer_classification/data/new_test_seqs_lbster_embeddings.parquet', engine='pyarrow', index=False)


# COMMAND ----------

# MAGIC %md
# MAGIC # AbLang2

# COMMAND ----------

heavy_chains = df["vh"].to_list()
light_chains = df["vl"].to_list()
print(len(heavy_chains))
print(len(light_chains))

# COMMAND ----------

# prep data format for ablang2
sequences = []
for i in range(len(heavy_chains)):
    vh = heavy_chains[i]
    vl = light_chains[i]
    seq = [vh, vl]
    sequences.append(seq)
print(len(sequences))
sequences[0]

# COMMAND ----------

import ablang2

# Download and initialise the model
ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device='cpu')

embeddings = ablang(sequences, mode='seqcoding')
print(type(embeddings))
print(len(embeddings))

# COMMAND ----------

print(embeddings.shape)

# COMMAND ----------

# saving the combined embeddings
# embeddings = [emb for emb in embeddings]

# Stack the arrays to create a single array
embeddings_array = np.vstack(embeddings)
print(embeddings_array.shape)

emb_df = pd.DataFrame(embeddings_array)
print(emb_df.shape)
cols = ["ablang2_" + str(x) for x in emb_df.columns]
emb_df.columns = cols

de = pd.concat([df[["antibody_id", "antibody_name"]], emb_df], axis=1)
print(de.shape)
de.head(2)


# COMMAND ----------

de.to_parquet('gdpa1_ablang2_embeddings.parquet', engine='pyarrow', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # PoET embeddings

# COMMAND ----------

!pip install openprotein-python


# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.special
import json

import openprotein
import numpy as np
import getpass

# COMMAND ----------

# Get a model instance by name:
import openprotein
session = openprotein.connect(username="user", password="password")
# list available models:
print(session.embedding.list_models() )
# init model by name
model = session.embedding.get_model('prot-seq')

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Saved Embeddings

# COMMAND ----------

import umap
import plotly.express as px

# COMMAND ----------

df.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ablang2 umap

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

umap_model = umap.UMAP(
    n_components=2,        
    n_neighbors=15,        
    min_dist=0.1,          
    metric='euclidean',   
    random_state=42,      
    verbose=False
)

# COMMAND ----------

emb_type = "ablang2"

embeddings_path = "gdpa1_ablang2_embeddings.parquet"
embeddings = pd.read_parquet(embeddings_path)
print(embeddings.shape)

scaler = StandardScaler()
feature_cols = [x for x in embeddings.columns if x not in ["antibody_id", "antibody_name"]]
X_scaled = scaler.fit_transform(embeddings[feature_cols].values)


umap_model = umap.UMAP(
    n_components=2,        
    n_neighbors=15,        
    min_dist=0.1,          
    metric='euclidean',   
    random_state=42,      
    verbose=False
)
embedding_umap = umap_model.fit_transform(X_scaled)

df[emb_type+"_umap1"] = embedding_umap[:, 0]
df[emb_type+"_umap2"] = embedding_umap[:, 1]

df.head(2)

# COMMAND ----------

px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="lc_subtype")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ESM2 umaps

# COMMAND ----------

emb_type = "esm2_mean_no_sep"

embeddings_path = "gdpa1_esm2_mean_no_sep_embeddings.parquet"
embeddings = pd.read_parquet(embeddings_path)
print(embeddings.shape)

scaler = StandardScaler()
feature_cols = [x for x in embeddings.columns if x not in ["antibody_id", "antibody_name"]]
X_scaled = scaler.fit_transform(embeddings[feature_cols].values)


umap_model = umap.UMAP(
    n_components=2,        
    n_neighbors=15,        
    min_dist=0.1,          
    metric='euclidean',   
    random_state=42,      
    verbose=False
)
embedding_umap = umap_model.fit_transform(X_scaled)

df[emb_type+"_umap1"] = embedding_umap[:, 0]
df[emb_type+"_umap2"] = embedding_umap[:, 1]

df.head(2)

# px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="lc_subtype")

# COMMAND ----------

px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="hc_subtype")

# COMMAND ----------

emb_type = "esm2_cls"

embeddings_path = "gdpa1_esm2_cls_embeddings.parquet"
embeddings = pd.read_parquet(embeddings_path)
print(embeddings.shape)

scaler = StandardScaler()
feature_cols = [x for x in embeddings.columns if x not in ["antibody_id", "antibody_name"]]
X_scaled = scaler.fit_transform(embeddings[feature_cols].values)


umap_model = umap.UMAP(
    n_components=2,        
    n_neighbors=15,        
    min_dist=0.1,          
    metric='euclidean',   
    random_state=42,      
    verbose=False
)
embedding_umap = umap_model.fit_transform(X_scaled)

df[emb_type+"_umap1"] = embedding_umap[:, 0]
df[emb_type+"_umap2"] = embedding_umap[:, 1]

df.head(2)

# px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="lc_subtype")

# COMMAND ----------

px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="hc_subtype")

# COMMAND ----------

# MAGIC %md
# MAGIC ### IgBERT umap

# COMMAND ----------

emb_type = "igbert"

embeddings_path = "gdpa1_igbert_mean_embeddings.parquet"
embeddings = pd.read_parquet(embeddings_path)
print(embeddings.shape)

scaler = StandardScaler()
feature_cols = [x for x in embeddings.columns if x not in ["antibody_id", "antibody_name"]]
X_scaled = scaler.fit_transform(embeddings[feature_cols].values)


umap_model = umap.UMAP(
    n_components=2,        
    n_neighbors=15,        
    min_dist=0.1,          
    metric='euclidean',   
    random_state=42,      
    verbose=False
)
embedding_umap = umap_model.fit_transform(X_scaled)

df[emb_type+"_umap1"] = embedding_umap[:, 0]
df[emb_type+"_umap2"] = embedding_umap[:, 1]

df.head(2)

# px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="lc_subtype")

# COMMAND ----------

px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="hc_subtype")

# COMMAND ----------

# MAGIC %md
# MAGIC ### antiberta2 umap

# COMMAND ----------

emb_type = "antiberta2"

embeddings_path = "gdpa1_antiberta2_embeddings.parquet"
embeddings = pd.read_parquet(embeddings_path)
print(embeddings.shape)

scaler = StandardScaler()
feature_cols = [x for x in embeddings.columns if x not in ["antibody_id", "antibody_name"]]
X_scaled = scaler.fit_transform(embeddings[feature_cols].values)


umap_model = umap.UMAP(
    n_components=2,        
    n_neighbors=15,        
    min_dist=0.1,          
    metric='euclidean',   
    random_state=42,      
    verbose=False
)
embedding_umap = umap_model.fit_transform(X_scaled)

df[emb_type+"_umap1"] = embedding_umap[:, 0]
df[emb_type+"_umap2"] = embedding_umap[:, 1]

df.head(2)

# px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="lc_subtype")

# COMMAND ----------

px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="hc_subtype")

# COMMAND ----------

# MAGIC %md
# MAGIC ### antiberta2 cssp umap

# COMMAND ----------

emb_type = "antiberta2_cssp"

embeddings_path = "gdpa1_antiberta2_cssp_embeddings.parquet"
embeddings = pd.read_parquet(embeddings_path)
print(embeddings.shape)

scaler = StandardScaler()
feature_cols = [x for x in embeddings.columns if x not in ["antibody_id", "antibody_name"]]
X_scaled = scaler.fit_transform(embeddings[feature_cols].values)


umap_model = umap.UMAP(
    n_components=2,        
    n_neighbors=15,        
    min_dist=0.1,          
    metric='euclidean',   
    random_state=42,      
    verbose=False
)
embedding_umap = umap_model.fit_transform(X_scaled)

df[emb_type+"_umap1"] = embedding_umap[:, 0]
df[emb_type+"_umap2"] = embedding_umap[:, 1]

df.head(2)

# px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="lc_subtype")

# COMMAND ----------

px.scatter(data_frame=df, x=f"{emb_type}_umap1", y=f"{emb_type}_umap2", title=f"{emb_type} UMAP", color="hc_subtype")