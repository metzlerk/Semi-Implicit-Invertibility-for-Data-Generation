#!/usr/bin/env python3
"""
Script to test the unitary generator with the fixed embeddings loading.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the unitary generator
sys.path.append('/home/kjmetzler/ChemicalDataGeneration/models')
from unitary_generator import UnitaryDataGenerator

def main():
    # Default paths
    model_path = '/home/kjmetzler/scratch/trained_models/best_unitary_autoencoder.pth'
    data_path = '/home/kjmetzler/scratch/train_data.feather'
    embeddings_path = '/home/kjmetzler/ChemicalDataGeneration/name_smiles_embedding_file.csv'
    output_dir = '/home/kjmetzler/scratch/unitary_test'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create generator
    generator = UnitaryDataGenerator(
        model_path=model_path,
        data_path=data_path,
        embeddings_path=embeddings_path,
        batch_size=16,
        chunk_size=500
    )
    
    # Load components
    if not generator.load_model():
        logger.error("Failed to load model")
        return False
    
    if not generator.load_data():
        logger.error("Failed to load data")
        return False
    
    if not generator.load_embeddings():
        logger.error("Failed to load embeddings")
        return False
    
    # Check what chemicals are available in the embeddings
    logger.info("Available chemicals in embeddings:")
    for chem in generator.chemical_embeddings:
        logger.info(f"  {chem}: {len(generator.chemical_embeddings[chem])} dims")
    
    # Generate a few samples for each chemical to test
    for chem in ["DEB", "DEM", "DMMP", "DPM", "DtBP", "JP8", "MES", "TEPO"]:
        if chem in generator.chemical_embeddings:
            logger.info(f"Generating test samples for {chem}")
            samples = generator.generate_synthetic_samples(chem, 10)
            if samples is not None:
                logger.info(f"Generated {len(samples)} samples for {chem}")
                # Save samples
                samples.to_csv(f"{output_dir}/test_{chem}.csv", index=False)
            else:
                logger.error(f"Failed to generate samples for {chem}")
        else:
            logger.error(f"{chem} not found in embeddings")
    
    # Generate chunked data for all chemicals
    logger.info("Generating chunked data for all chemicals")
    success = generator.generate_chunked(
        total_samples=800,  # 100 per chemical 
        output_dir=output_dir,
        target_chemicals=["DEB", "DEM", "DMMP", "DPM", "DtBP", "JP8", "MES", "TEPO"]
    )
    
    if success:
        logger.info("Chunked data generation successful")
        
        # Check the output file
        output_file = os.path.join(output_dir, 'synthetic_data_chunked.feather')
        if os.path.exists(output_file):
            try:
                df = pd.read_feather(output_file)
                logger.info(f"Output file shape: {df.shape}")
                
                # Check if it has Label column
                if 'Label' in df.columns:
                    label_counts = df['Label'].value_counts().to_dict()
                    logger.info("Label counts:")
                    for label, count in label_counts.items():
                        logger.info(f"  {label}: {count}")
                
                # Check if it has one-hot encoded columns
                for chem in ["DEB", "DEM", "DMMP", "DPM", "DtBP", "JP8", "MES", "TEPO"]:
                    if chem in df.columns:
                        count = df[chem].sum()
                        logger.info(f"  {chem} (one-hot): {count}")
            except Exception as e:
                logger.error(f"Error checking output file: {e}")
        else:
            logger.error(f"Output file not found: {output_file}")
    else:
        logger.error("Chunked data generation failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
