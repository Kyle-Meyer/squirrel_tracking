from src.downloader import ImageDownloader
from src.squirrel_trainer import SquirrelYOLOTrainer
from pathlib import Path
import os

def find_latest_model(project_path="yolo_squirrel_project"):
    """Find the latest trained model"""
    runs_path = Path(project_path) / "runs" 
    
    if not runs_path.exists():
        print("No previous training runs found.")
        return None
    
    # Look for training directories
    train_dirs = [d for d in runs_path.iterdir() if d.is_dir() and "squirrel_detection" in d.name]
    
    if not train_dirs:
        print("No squirrel detection training runs found.")
        return None
    
    # Sort by modification time to get the latest
    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    
    # Look for best.pt and last.pt
    best_model = latest_dir / "weights" / "best.pt"
    last_model = latest_dir / "weights" / "last.pt"
    
    if best_model.exists():
        print(f"Found best model: {best_model}")
        return str(best_model)
    elif last_model.exists():
        print(f"Found last model: {last_model}")
        return str(last_model)
    else:
        print(f"No model weights found in {latest_dir}")
        return None

def evaluate_model_performance(trainer, model_path=None):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("🔍 COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Check annotation status first
    print("\n📊 Dataset Statistics:")
    stats = trainer.checkAnnotations()
    total_annotated = sum(s['annotated'] for s in stats.values())
    total_images = sum(s['total'] for s in stats.values())
    
    print(f"Total images: {total_images}")
    print(f"Annotated images: {total_annotated}")
    print(f"Annotation coverage: {total_annotated/total_images*100:.1f}%")
    
    if total_annotated == 0:
        print("❌ No annotated images found! Cannot evaluate model.")
        return None
    
    # Run validation on test set
    print("\n🎯 Running Model Validation...")
    try:
        results = trainer.evaluateModel(model_path)
        
        print("\n📈 Model Performance Metrics:")
        if hasattr(results, 'box'):
            metrics = results.box
            print(f"  • mAP50: {metrics.map50:.3f} (mean Average Precision at IoU=0.5)")
            print(f"  • mAP50-95: {metrics.map:.3f} (mAP across IoU 0.5-0.95)")
            print(f"  • Precision: {metrics.mp:.3f}")
            print(f"  • Recall: {metrics.mr:.3f}")
            
            # Performance interpretation
            print("\n🎯 Performance Interpretation:")
            if metrics.map50 > 0.8:
                print("  ✅ Excellent performance! Model is detecting squirrels very well.")
            elif metrics.map50 > 0.6:
                print("  ✅ Good performance! Model is working well with room for improvement.")
            elif metrics.map50 > 0.4:
                print("  ⚠️  Moderate performance. Consider more training or data.")
            else:
                print("  ❌ Poor performance. More training data or different approach needed.")
                
        else:
            print("  ⚠️  Detailed metrics not available, but validation completed.")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None
    
    return results

def test_sample_predictions(trainer, model_path=None):
    """Test model on sample images"""
    print("\n" + "="*60)
    print("🖼️  SAMPLE PREDICTIONS TEST")
    print("="*60)
    
    # Get some test images
    test_images_dir = Path("yolo_squirrel_project/dataset/test/images")
    
    if not test_images_dir.exists():
        print("❌ No test images directory found.")
        return
    
    test_images = list(test_images_dir.glob("*.jpg"))[:5]  # Test on first 5 images
    
    if not test_images:
        print("❌ No test images found.")
        return
    
    print(f"Testing on {len(test_images)} sample images...")
    
    try:
        for i, img_path in enumerate(test_images):
            print(f"\n🔍 Testing image {i+1}: {img_path.name}")
            results = trainer.predictSample(str(img_path), model_path, aSaveResults=True)
            
            # Analyze results
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    num_detections = len(result.boxes)
                    if num_detections > 0:
                        confidences = result.boxes.conf.cpu().numpy()
                        max_conf = max(confidences)
                        avg_conf = sum(confidences) / len(confidences)
                        print(f"  ✅ Found {num_detections} squirrel(s)")
                        print(f"  📊 Confidence - Max: {max_conf:.3f}, Avg: {avg_conf:.3f}")
                    else:
                        print("  ❌ No squirrels detected")
                else:
                    print("  ❌ No detections")
            else:
                print("  ❌ Prediction failed")
                
    except Exception as e:
        print(f"❌ Sample prediction failed: {e}")

def main():
    print("🐿️  SQUIRREL DETECTION - CONTINUED TRAINING & EVALUATION")
    print("="*70)
    
    # Initialize trainer
    trainer = SquirrelYOLOTrainer(
        aDataSetPath="resources/squirrel_base_images",
        aModelVariant="yolov8n.pt"  # Will be overridden if continuing training
    )
    
    # Check for existing models
    latest_model = find_latest_model()
    
    print("\n📋 TRAINING OPTIONS:")
    print("1. Continue training from latest model")
    print("2. Start fresh training")
    print("3. Evaluate existing model only")
    print("4. Test sample predictions")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if latest_model:
            print(f"\n🔄 Continuing training from: {latest_model}")
            # Load the existing model
            trainer.mModelVariant = latest_model
            trainer.loadModel()
            
            # Continue training with more epochs
            print("\n🏋️  Starting continued training...")
            results = trainer.trainModel(
                epochs=50,           # Additional epochs
                imgsz=640,
                batch=8,             # Adjust based on your GPU memory
                device='cpu',       # 'cuda' if you have GPU, 'cpu' otherwise
                patience=20,         # Early stopping patience
                save_period=10       # Save checkpoint every 10 epochs
            )
            print("✅ Continued training completed!")
            
            # Evaluate the updated model
            evaluate_model_performance(trainer)
            
        else:
            print("❌ No existing model found. Starting fresh training...")
            choice = "2"
    
    if choice == "2":
        print("\n🆕 Starting fresh training...")
        
        # Check annotations first
        stats = trainer.checkAnnotations()
        total_annotated = sum(s['annotated'] for s in stats.values())
        
        if total_annotated == 0:
            print("❌ No annotated images found! Please annotate your images first.")
            trainer.getAnnotationGuide()
            return
            
        print(f"✅ Found {total_annotated} annotated images. Starting training...")
        
        results = trainer.trainModel(
            epochs=100,          # Full training epochs
            imgsz=640,
            batch=8,
            device='cpu',
            patience=30
        )
        print("✅ Training completed!")
        
        # Evaluate the new model
        evaluate_model_performance(trainer)
    
    elif choice == "3":
        if latest_model:
            print(f"\n📊 Evaluating model: {latest_model}")
            evaluate_model_performance(trainer, latest_model)
            test_sample_predictions(trainer, latest_model)
        else:
            print("❌ No trained model found to evaluate.")
    
    elif choice == "4":
        if latest_model:
            print(f"\n🧪 Testing predictions with: {latest_model}")
            test_sample_predictions(trainer, latest_model)
        else:
            print("❌ No trained model found for testing.")
    
    else:
        print("❌ Invalid choice. Please run again and select 1-4.")
    
    print("\n" + "="*70)
    print("🎉 SQUIRREL DETECTION SESSION COMPLETE!")
    print("="*70)
    
    # Show where results are saved
    print("\n📁 Results saved in:")
    print(f"  • Training runs: yolo_squirrel_project/runs/detect/")
    print(f"  • Model weights: yolo_squirrel_project/runs/detect/squirrel_detection*/weights/")
    print(f"  • Predictions: yolo_squirrel_project/runs/detect/predictions*/")

if __name__ == "__main__":
    main()
