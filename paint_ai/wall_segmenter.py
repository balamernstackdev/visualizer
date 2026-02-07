import numpy as np
import cv2
import streamlit as st
from .sam_loader import get_mask_generator

class WallSegmenter:
    def __init__(self, sam_model):
        self.sam = sam_model
        self.mask_generator = get_mask_generator(sam_model)

    def detect_potential_walls(self, image_np):
        """
        Runs automatic mask generation and filters for wall-like regions.
        Returns a list of masks (dict with 'segmentation', 'area', etc.)
        """
        # SAM expects RGB 0-255
        masks = self.mask_generator.generate(image_np)
        
        # Heuristic filtering for "walls":
        # Walls are usually large.
        filtered_masks = []
        image_area = image_np.shape[0] * image_np.shape[1]
        min_area = 20 # Absolute 20 pixels - pretty much anything visible
        
        for mask in masks:
            if mask['area'] > min_area:
                filtered_masks.append(mask)
        
        # Sort by area descending (largest walls first usually better)
        filtered_masks.sort(key=lambda x: x['area'], reverse=True)
        return filtered_masks

    @staticmethod
    def get_mask_by_point(masks, x, y):
        """
        Finds the smallest mask containing the point (x, y).
        Strategy: Check all masks, find those containing point.
        Among those, prefer the *smallest* one (most specific region),
        or the one with highest stability score?
        Usually, smallest area containing point = specific object.
        """
        candidates = []
        for i, mask in enumerate(masks):
            # mask['segmentation'] is boolean array
            if mask['segmentation'][y, x]:
                candidates.append(mask)
        
        if not candidates:
            return None
            
        # Return smallest candidate (most specific)
        return min(candidates, key=lambda x: x['area'])
