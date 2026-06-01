#!/usr/bin/env python3
"""
Generate GeneCompass Embeddings for Single-Cell Data

This script generates embeddings for single-cell data using the pretrained
GeneCompass model.
"""

from datasets import load_from_disk
import torch
from tqdm import tqdm
import pickle
import os
import gc
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_batch_optimized(batch_indices, dataset, model, device):
    """Process a batch of data with optimized memory usage"""
    with torch.no_grad():
        # Get all data at once
        start_idx, end_idx = batch_indices
        input_ids = torch.tensor(dataset['input_ids'][start_idx:end_idx]).to(device)
        values = torch.tensor(dataset['values'][start_idx:end_idx]).to(device)
        species = torch.tensor(dataset['species'][start_idx:end_idx]).to(device)

        # Forward pass
        emb = model.bert.forward(input_ids=input_ids, values=values, species=species)[0]
        emb = emb[:, 1:, :]  # Remove first token

        # Move to CPU and convert to numpy to free GPU memory
        emb_cpu = emb.cpu().numpy()

        # Clean GPU memory
        del input_ids, values, species, emb
        torch.cuda.empty_cache()

        return emb_cpu


def generate_embeddings(dataset_path, model_path, token_dict_path, output_path, 
                       batch_size=128, gpu_ids="0"):
    """
    Generate embeddings for single-cell dataset
    
    Args:
        dataset_path: Path to the preprocessed dataset
        model_path: Path to the pretrained GeneCompass model
        token_dict_path: Path to token dictionary
        output_path: Path to save the embeddings
        batch_size: Batch size for processing
        gpu_ids: GPU IDs to use (e.g., "0,1,2,3")
    """
    # Set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    logger.info(f"Loading token dictionary from {token_dict_path}")
    with open(token_dict_path, "rb") as fp:
        token_dictionary = pickle.load(fp)
    
    # Load knowledge embeddings
    logger.info("Loading knowledge embeddings...")
    knowledges = dict()
    
    try:
        from genecompass.utils import load_prior_embedding
        out = load_prior_embedding(token_dictionary_or_path=token_dict_path)
        
        knowledges['promoter'] = out[0]
        knowledges['co_exp'] = out[1]
        knowledges['gene_family'] = out[2]
        knowledges['peca_grn'] = out[3]
        knowledges['homologous_gene_human2mouse'] = out[4]
        logger.info("Knowledge embeddings loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load knowledge embeddings: {e}")
        logger.info("Continuing without knowledge embeddings...")
        knowledges = None
    
    # Load dataset and model
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    logger.info(f"Loading model from {model_path}")
    try:
        from genecompass.modeling_bert import BertForMaskedLM
        model = BertForMaskedLM.from_pretrained(
            model_path,
            knowledges=knowledges if knowledges else None,
        )
    except Exception as e:
        logger.error(f"Failed to load GeneCompass model: {e}")
        raise
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    logger.info(f"Using device: {device}")
    
    # Memory optimization
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    # Main processing loop
    total_length = len(dataset)
    iters = total_length // batch_size
    
    total_batches = iters if total_length % batch_size == 0 else iters + 1
    logger.info(f"Starting processing: Total samples: {total_length}, "
                f"Batch size: {batch_size}, Total iterations: {total_batches}")
    
    # Use list to store results on CPU
    emb_list = []
    
    with torch.no_grad():
        for i in tqdm(range(total_batches), desc="Processing batches"):
            try:
                # Calculate current batch indices
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_length)
                
                # Process current batch
                emb_batch = process_batch_optimized(
                    (start_idx, end_idx), dataset, model, device
                )
                emb_list.append(emb_batch)
                
                # Periodic cleanup (every 10 batches)
                if i % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"Batch {i} ran out of GPU memory. Try reducing batch size.")
                    raise e
                else:
                    raise e
    
    # Save results
    logger.info(f"Saving results to {output_path}")
    
    # Concatenate all results
    emb_numpy = np.concatenate(emb_list, axis=0)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(emb_numpy, f)
    
    logger.info(f"Processing completed! Embeddings shape: {emb_numpy.shape}")
    logger.info(f"Embeddings saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate GeneCompass embeddings')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the preprocessed dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained GeneCompass model')
    parser.add_argument('--token_dict_path', type=str, required=True,
                        help='Path to token dictionary')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the embeddings')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for processing')
    parser.add_argument('--gpu_ids', type=str, default="0",
                        help='GPU IDs to use (e.g., "0,1,2,3")')
    
    args = parser.parse_args()
    
    generate_embeddings(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        token_dict_path=args.token_dict_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        gpu_ids=args.gpu_ids
    )
