# train_classifier_from_images.py
import os
import json
import pickle
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import MMDetection and MMPose
try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
    from mmpose.datasets import DatasetInfo
    from mmdet.apis import process_mmdet_results
    MM_AVAILABLE = True
except ImportError:
    print("Warning: MMDetection or MMPose not available. Using fallback mode.")
    MM_AVAILABLE = False

class BJJImageClassifier:
    def __init__(self):
        self.det_model = None
        self.pose_model = None
        self.classifier = None
        self.dataset_info = None
        
    def load_models(self, device='cuda:0'):
        """Load detection and pose estimation models"""
        if not MM_AVAILABLE:
            print("MMDetection/MMPose not available. Using fallback mode.")
            return False
            
        try:
            # Initialize detection model
            self.det_model = init_detector(
                'mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py',
                'checkpoints/detection/deformable_detr_twostage_refine.pth',
                device=device
            )
            
            # Initialize pose model
            self.pose_model = init_pose_model(
                'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py',
                'checkpoints/vitpose-h-multi-coco.pth',
                device=device
            )
            
            # Setup dataset info
            dataset = self.pose_model.cfg.data['test']['type']
            dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
            self.dataset_info = DatasetInfo(dataset_info)
            
            print("✓ Detection and pose models loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False
    
    def extract_poses_from_image(self, image_path):
        """Extract poses from a single image using MMDetection/MMPose"""
        if not MM_AVAILABLE or self.det_model is None:
            return None, None
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return None, None
            
            # Detect people
            mmdet_results = inference_detector(self.det_model, img)
            person_results = process_mmdet_results(mmdet_results, 1)  # person class
            
            if len(person_results) < 2:
                return None, None  # Need exactly 2 people for BJJ
            
            # Get poses
            pose_results, _ = inference_top_down_pose_model(
                self.pose_model,
                img,
                person_results,
                bbox_thr=0.3,
                format='xyxy',
                dataset_info=self.dataset_info
            )
            
            if len(pose_results) >= 2:
                # Return the two most confident detections
                pose_results.sort(key=lambda x: x['bbox'][4], reverse=True)
                return pose_results[0]['keypoints'], pose_results[1]['keypoints']
            
            return None, None
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None
    
    def normalize_poses(self, kpts1, kpts2):
        """Normalize poses to be scale and translation invariant"""
        if kpts1 is None or kpts2 is None:
            return None, None
            
        # Combine keypoints
        coords = np.concatenate([kpts1, kpts2])
        
        # Extract coordinates (ignore confidence scores for normalization)
        coordinates = coords[:, :2]
        confidences = coords[:, 2]
        
        # Normalize coordinates
        xmin, ymin = coordinates.min(axis=0)
        xmax, ymax = coordinates.max(axis=0)
        
        # Center and scale
        coordinates[:, 0] = coordinates[:, 0] - xmin
        coordinates[:, 1] = coordinates[:, 1] - ymin
        
        max_dim = max(xmax - xmin, ymax - ymin)
        if max_dim > 0:
            coordinates = coordinates / max_dim
        
        # Recombine with confidence scores
        kpts1_norm = np.column_stack([coordinates[:17], confidences[:17]])
        kpts2_norm = np.column_stack([coordinates[17:], confidences[17:]])
        
        return kpts1_norm, kpts2_norm
    
    def prepare_dataset_split(self, annotations_json, images_dir, split_name="dataset", max_samples=None):
        """Prepare data from a single split (train/val/test)"""
        print(f"\n=== Preparing {split_name} data ===")
        print(f"Annotations: {annotations_json}")
        print(f"Images dir: {images_dir}")
        
        with open(annotations_json, 'r') as f:
            coco_data = json.load(f)
        
        X = []  # Features
        y = []  # Labels
        skipped_no_people = 0
        skipped_other = 0
        processed = 0
        
        # Create mappings
        image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
        image_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            image_annotations[ann['image_id']].append(ann)
        category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        total_images = len(image_annotations)
        print(f"Found {total_images} annotated images")
        
        for img_id, annotations in image_annotations.items():
            if max_samples and len(X) >= max_samples:
                break
                
            if img_id not in image_id_to_file:
                continue
                
            image_file = image_id_to_file[img_id]
            image_path = os.path.join(images_dir, image_file)
            
            if not os.path.exists(image_path):
                skipped_other += 1
                continue
            
            # Get position label
            category_id = annotations[0]['category_id']
            position_label = category_id_to_name[category_id]
            
            # Skip transition frames for training
            if position_label == 'transition':
                skipped_other += 1
                continue
            
            # Extract poses
            kpts1, kpts2 = self.extract_poses_from_image(image_path)
            
            if kpts1 is not None and kpts2 is not None:
                # Normalize poses
                kpts1_norm, kpts2_norm = self.normalize_poses(kpts1, kpts2)
                
                if kpts1_norm is not None and kpts2_norm is not None:
                    # Create feature vector
                    features = kpts1_norm.flatten().tolist() + kpts2_norm.flatten().tolist()
                    
                    X.append(features)
                    y.append(position_label)
                    processed += 1
                else:
                    skipped_no_people += 1
            else:
                skipped_no_people += 1
            
            # Progress reporting
            if processed % 100 == 0 and processed > 0:
                print(f"  Processed {processed} images...")
        
        print(f"✓ {split_name}: {len(X)} usable samples")
        print(f"  Skipped (no people): {skipped_no_people}")
        print(f"  Skipped (other): {skipped_other}")
        
        # Class distribution
        if y:
            from collections import Counter
            class_dist = Counter(y)
            print(f"  Class distribution: {dict(class_dist)}")
        
        return np.array(X), np.array(y)
    
    def train_classifier(self, X_train, y_train, X_val, y_val, random_state=42):
        """Train the MLP classifier with validation set"""
        if len(X_train) == 0:
            print("Error: No training data available!")
            return 0.0
        
        print(f"\n=== Training Classifier ===")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        # Train classifier with validation
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            verbose=True
        )
        
        print("Starting training...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_accuracy = self.evaluate(X_val, y_val, "Validation")
        
        return val_accuracy
    
    def evaluate(self, X, y, split_name="Dataset"):
        """Evaluate classifier on a dataset split"""
        if self.classifier is None:
            print("Error: No classifier available for evaluation!")
            return 0.0
        
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"\n=== {split_name} Evaluation ===")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Samples: {len(X)}")
        
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, zero_division=0))
        
        # Show some predictions
        print(f"\nExample predictions (first 10):")
        for i in range(min(10, len(X))):
            status = "✓" if y[i] == y_pred[i] else "✗"
            print(f"  {status} True: {y[i]:<20} Predicted: {y_pred[i]}")
        
        return accuracy
    
    def save_classifier(self, output_path):
        """Save trained classifier"""
        if self.classifier is None:
            print("Error: No classifier to save! Train first.")
            return False
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"✓ Classifier saved to: {output_path}")
        return True
    
    def load_classifier(self, classifier_path):
        """Load a pre-trained classifier"""
        try:
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            print(f"✓ Classifier loaded from: {classifier_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading classifier: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Train BJJ Position Classifier with Train/Val/Test Splits')
    
    # Data arguments
    parser.add_argument('--train-annotations', type=str, required=True,
                       help='Path to training annotations JSON file')
    parser.add_argument('--train-images-dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--val-annotations', type=str, required=True,
                       help='Path to validation annotations JSON file')
    parser.add_argument('--val-images-dir', type=str, required=True,
                       help='Directory containing validation images')
    parser.add_argument('--test-annotations', type=str,
                       help='Path to test annotations JSON file (optional)')
    parser.add_argument('--test-images-dir', type=str,
                       help='Directory containing test images (optional)')
    
    # Training arguments
    parser.add_argument('--output', type=str, default='checkpoints/jiujitsu/classifier.pickle',
                       help='Output path for trained classifier')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples per split (for testing)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0 or cpu)')
    
    # Evaluation mode
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate, do not train')
    parser.add_argument('--classifier', type=str,
                       help='Path to existing classifier for evaluation')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BJJImageClassifier()
    
    if args.eval_only:
        # Evaluation mode only
        if not args.classifier:
            print("Error: --classifier required for evaluation mode")
            return
            
        if trainer.load_classifier(args.classifier):
            # Evaluate on all provided splits
            if args.train_annotations and args.train_images_dir:
                X_train, y_train = trainer.prepare_dataset_split(
                    args.train_annotations, args.train_images_dir, "TRAIN", args.max_samples
                )
                if len(X_train) > 0:
                    trainer.evaluate(X_train, y_train, "TRAIN")
            
            if args.val_annotations and args.val_images_dir:
                X_val, y_val = trainer.prepare_dataset_split(
                    args.val_annotations, args.val_images_dir, "VALIDATION", args.max_samples
                )
                if len(X_val) > 0:
                    trainer.evaluate(X_val, y_val, "VALIDATION")
            
            if args.test_annotations and args.test_images_dir:
                X_test, y_test = trainer.prepare_dataset_split(
                    args.test_annotations, args.test_images_dir, "TEST", args.max_samples
                )
                if len(X_test) > 0:
                    trainer.evaluate(X_test, y_test, "TEST")
    
    else:
        # Training mode
        print("Loading detection and pose models...")
        models_loaded = trainer.load_models(device=args.device)
        
        if not models_loaded:
            print("Cannot proceed without pose estimation models.")
            return
        
        # Prepare all splits
        X_train, y_train = trainer.prepare_dataset_split(
            args.train_annotations, args.train_images_dir, "TRAIN", args.max_samples
        )
        
        X_val, y_val = trainer.prepare_dataset_split(
            args.val_annotations, args.val_images_dir, "VALIDATION", args.max_samples
        )
        
        # Train classifier
        if len(X_train) > 0 and len(X_val) > 0:
            print("\n=== Starting Training ===")
            val_accuracy = trainer.train_classifier(X_train, y_train, X_val, y_val)
            
            # Save classifier
            if val_accuracy > 0:
                trainer.save_classifier(args.output)
                
                # Evaluate on test set if provided
                if args.test_annotations and args.test_images_dir:
                    X_test, y_test = trainer.prepare_dataset_split(
                        args.test_annotations, args.test_images_dir, "TEST", args.max_samples
                    )
                    if len(X_test) > 0:
                        test_accuracy = trainer.evaluate(X_test, y_test, "TEST")
                        print(f"\n🎯 FINAL RESULTS:")
                        print(f"Validation Accuracy: {val_accuracy:.3f}")
                        print(f"Test Accuracy: {test_accuracy:.3f}")
        else:
            print("Error: Not enough data for training!")

if __name__ == "__main__":
    main()