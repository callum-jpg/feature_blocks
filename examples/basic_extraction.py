#!/usr/bin/env python3
"""
Basic feature extraction example.

This example demonstrates how to extract features from an image using
the Feature Blocks package with the simplest configuration.
"""

import numpy 
from feature_blocks.features import extract
from feature_blocks.models import UNI

def main():
    # Create a dummy image for demonstration
    # In practice, you would load your actual image
    dummy_image = numpy.random.randint(0, 255, size=(3, 1000, 1000), dtype=numpy.uint8)
    
    # Initialize the UNI model
    print("Loading UNI model...")
    model = UNI()
    
    # Extract features from image blocks
    print("Extracting features...")
    features = extract(
        image=dummy_image,
        model=model,
        block_size=112,
        block_method="block"
    )
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Feature dimensionality: {model.n_features}")
    
    # Save features (in practice, you might want to save to zarr)
    numpy.save("extracted_features.npy", features)
    print("Features saved to 'extracted_features.npy'")

if __name__ == "__main__":
    main()