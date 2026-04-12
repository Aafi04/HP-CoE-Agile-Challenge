"""
Confidence Calibration for Fine-tuned Kaggle Model
===================================================
The fine-tuned model on Kaggle domain requires different confidence calibration
than the pre-trained model on FaceForensics++.

This module provides:
1. Domain-specific calibration formulas
2. Temperature scaling
3. Threshold optimization
"""

import numpy as np
import torch
from scipy import stats


class ConfidenceCalibrator:
    """
    Calibrate raw model output to interpretable confidence scores
    """
    
    def __init__(self, domain='kaggle'):
        """
        domain: 'kaggle' (fine-tuned) or 'faceforensics' (pre-trained)
        """
        self.domain = domain
        
        # Domain-specific parameters (learned from validation sets)
        if domain == 'kaggle':
            # Fine-tuned model on Kaggle natural images
            self.method = 'shift'
            self.params = {
                'scale': 1.3,
                'offset': -0.08,
                'temperature': 1.0,
                'threshold': 0.3
            }
            self.description = "Kaggle fine-tuned model calibration"
        
        elif domain == 'faceforensics':
            # Pre-trained on FaceForensics++ compressed videos
            self.method = 'shift'
            self.params = {
                'scale': 1.5,
                'offset': -0.1,
                'temperature': 1.0,
                'threshold': 0.5
            }
            self.description = "FaceForensics++ pre-trained model calibration"
        
        else:
            raise ValueError(f"Unknown domain: {domain}")
    
    def calibrate_raw_confidence(self, raw_conf):
        """
        Convert raw sigmoid output (0-1) to calibrated confidence
        
        raw_conf: float or np.array with values in [0, 1]
        returns: calibrated confidence in [0, 1]
        """
        raw_conf = np.asarray(raw_conf)
        
        if self.method == 'shift':
            # Shift method: conf * scale + offset
            calibrated = raw_conf * self.params['scale'] + self.params['offset']
        
        elif self.method == 'temperature':
            # Temperature scaling: 1 / (1 + exp(-T * (x - 0.5)))
            T = self.params['temperature']
            calibrated = 1.0 / (1.0 + np.exp(-T * (raw_conf - 0.5)))
        
        else:
            calibrated = raw_conf
        
        # Clip to [0, 1]
        calibrated = np.clip(calibrated, 0, 1)
        return calibrated
    
    def get_decision(self, raw_confidence):
        """
        Convert calibrated confidence to binary decision
        
        returns: (decision: bool, confidence: float)
                 where decision=True means DEEPFAKE
        """
        calibrated = self.calibrate_raw_confidence(raw_confidence)
        threshold = self.params['threshold']
        decision = calibrated > threshold
        return decision, calibrated
    
    def get_metrics(self, raw_confidence):
        """
        Get comprehensive metrics from raw confidence
        
        returns: dict with decision, confidence, risk level, etc.
        """
        calibrated = self.calibrate_raw_confidence(raw_confidence)
        decision = calibrated > self.params['threshold']
        
        # Risk level (how far from threshold)
        distance_from_threshold = abs(calibrated - self.params['threshold'])
        if distance_from_threshold < 0.1:
            risk_level = 'UNCERTAIN'
        elif distance_from_threshold < 0.2:
            risk_level = 'BORDERLINE'
        else:
            risk_level = 'CONFIDENT'
        
        return {
            'raw_confidence': float(raw_confidence),
            'calibrated_confidence': float(calibrated),
            'decision': 'DEEPFAKE' if decision else 'REAL',
            'decision_confidence': float(max(calibrated, 1 - calibrated)),
            'threshold': self.params['threshold'],
            'risk_level': risk_level,
            'domain': self.domain
        }


class ConfidenceValidator:
    """
    Validate and analyze confidence scores
    """
    
    @staticmethod
    def analyze_distribution(predictions, labels):
        """
        Analyze calibration: how well do confidence scores match accuracy?
        
        predictions: array of raw confidences in [0, 1]  
        labels: array of labels (0/1 for real/fake)
        """
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        
        # Separate by class
        fake_preds = predictions[labels == 1]
        real_preds = predictions[labels == 0]
        
        return {
            'fake_mean': float(np.mean(fake_preds)) if len(fake_preds) > 0 else None,
            'fake_std': float(np.std(fake_preds)) if len(fake_preds) > 0 else None,
            'real_mean': float(np.mean(real_preds)) if len(real_preds) > 0 else None,
            'real_std': float(np.std(real_preds)) if len(real_preds) > 0 else None,
            'separation': float(abs(np.mean(fake_preds) - np.mean(real_preds))) if len(fake_preds) > 0 and len(real_preds) > 0 else None
        }
    
    @staticmethod
    def find_optimal_threshold(predictions, labels):
        """Find threshold that maximizes F1 score"""
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.linspace(0, 1, 100):
            preds_binary = (predictions > threshold).astype(int)
            
            tp = np.sum((preds_binary == 1) & (labels == 1))
            fp = np.sum((preds_binary == 1) & (labels == 0))
            fn = np.sum((preds_binary == 0) & (labels == 1))
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    @staticmethod
    def calibration_error(predictions, labels, n_bins=10):
        """Calculate expected calibration error"""
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        total_samples = len(predictions)
        ece = 0
        
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(labels[mask])
                bin_conf = np.mean(predictions[mask])
                bin_weight = np.sum(mask) / total_samples
                ece += bin_weight * abs(bin_acc - bin_conf)
        
        return ece


# Convenience functions
def calibrate_kaggle(raw_confidence):
    """Quick calibration for Kaggle fine-tuned model"""
    cal = ConfidenceCalibrator('kaggle')
    return cal.calibrate_raw_confidence(raw_confidence)


def calibrate_faceforensics(raw_confidence):
    """Quick calibration for FaceForensics++ pre-trained model"""
    cal = ConfidenceCalibrator('faceforensics')
    return cal.calibrate_raw_confidence(raw_confidence)


def get_decision_kaggle(raw_confidence):
    """Get decision for Kaggle model"""
    cal = ConfidenceCalibrator('kaggle')
    decision, confidence = cal.get_decision(raw_confidence)
    return {
        'is_fake': decision,
        'confidence': confidence,
        'label': 'DEEPFAKE' if decision else 'REAL'
    }


if __name__ == "__main__":
    # Test calibration
    print("Kaggle Model Calibration Test")
    print("=" * 50)
    
    cal = ConfidenceCalibrator('kaggle')
    
    test_scores = [0.1, 0.25, 0.3, 0.5, 0.7, 0.9]
    
    for raw in test_scores:
        calibrated = cal.calibrate_raw_confidence(raw)
        decision, conf = cal.get_decision(raw)
        metrics = cal.get_metrics(raw)
        
        print(f"\nRaw: {raw:.2f} → Calibrated: {calibrated:.4f}")
        print(f"  Decision: {metrics['decision']} (confidence: {metrics['decision_confidence']:.1%})")
        print(f"  Risk Level: {metrics['risk_level']}")
