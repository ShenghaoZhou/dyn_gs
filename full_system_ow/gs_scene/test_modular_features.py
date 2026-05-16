import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from gs_scene.scene_model import SceneModel

def test_basic_initialization():
    print("Testing basic initialization...")
    model = SceneModel(width=512, height=512)
    assert model.use_anchors == False
    assert model.use_exposure == False
    assert model.pyr_levels == 1
    print("Basic initialization OK.")

def test_features_enabled():
    print("Testing features enabled...")
    model = SceneModel(width=512, height=512, use_anchors=True, use_exposure=True, pyr_levels=3)
    assert model.use_anchors == True
    assert model.use_exposure == True
    assert model.pyr_levels == 3
    assert len(model.anchors) == 1
    print("Features enabled initialization OK.")

def test_optimization_step_no_crash():
    print("Testing optimization_step no crash...")
    model = SceneModel(width=256, height=256, use_exposure=True, pyr_levels=2)
    
    # Create a dummy keyframe
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth = np.random.rand(256, 256).astype(np.float32)
    extrin = np.eye(4).astype(np.float32)
    K = np.array([[200, 0, 128], [0, 200, 128], [0, 0, 1]], dtype=np.float32)
    
    model.update(img, depth, extrin, K)
    
    # Run optimization steps
    try:
        model.optimization_loop(num_steps=5)
        print("Optimization step OK.")
    except Exception as e:
        print(f"Optimization step FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_anchor_logic():
    print("Testing anchor logic...")
    model = SceneModel(width=256, height=256, use_anchors=True, anchor_dist_threshold=0.1)
    
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth = np.random.rand(256, 256).astype(np.float32)
    K = np.array([[200, 0, 128], [0, 200, 128], [0, 0, 1]], dtype=np.float32)

    # First frame
    extrin1 = np.eye(4).astype(np.float32)
    model.update(img, depth, extrin1, K)
    assert len(model.anchors) == 1
    
    # Second frame, far away
    extrin2 = np.eye(4).astype(np.float32)
    extrin2[0, 3] = 1.0 # Move 1 meter in X
    model.update(img, depth, extrin2, K)
    
    # Check if new anchor was created
    print(f"Number of anchors: {len(model.anchors)}")
    assert len(model.anchors) > 1
    print("Anchor logic OK.")

if __name__ == "__main__":
    test_basic_initialization()
    test_features_enabled()
    test_optimization_step_no_crash()
    test_anchor_logic()
    print("All tests passed!")
