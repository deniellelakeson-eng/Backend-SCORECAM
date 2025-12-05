"""
HerbaScan Backend - Score-CAM Implementation
True score-weighted Class Activation Mapping using TensorFlow
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


class ScoreCAMGenerator:
    """
    Generates Score-CAM (Score-weighted Class Activation Mapping) visualizations.
    
    This provides true score-based explanations of CNN predictions, showing
    which regions of the input image were most important for the classification.
    Score-CAM uses forward pass confidence scores instead of gradients.
    """
    
    def __init__(self, model, layer_name: str = None):
        """
        Initialize Score-CAM generator.
        
        Args:
            model: Trained Keras model
            layer_name: Name of the last convolutional layer to use for Score-CAM.
                       If None or 'auto', will auto-detect the last conv layer.
        """
        self.model = model
        self.layer_name = layer_name if layer_name and layer_name != 'auto' else None
        self.score_model = self._create_score_model()
    
    def _create_score_model(self):
        """
        Create model that outputs both the target layer activations
        and the final predictions.
        Dynamically identifies the last convolutional layer for different model architectures.
        """
        # First, try to find the last convolutional layer automatically
        # This works for both MobileNetV2 and custom herbascan_model architectures
        target_layer = None
        last_conv_layer = None
        
        # Search for the last convolutional layer
        for layer in reversed(self.model.layers):
            # Check if it's a convolutional layer (Conv2D, DepthwiseConv2D, etc.)
            if isinstance(layer, (tf.keras.layers.Conv2D, 
                                  tf.keras.layers.DepthwiseConv2D,
                                  tf.keras.layers.SeparableConv2D)):
                last_conv_layer = layer
                break
        
        # If no conv layer found by type, try by name
        if last_conv_layer is None:
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower() or 'Conv' in layer.name:
                    last_conv_layer = layer
                    break
        
        # Use the found layer or try the specified layer_name
        if last_conv_layer is not None:
            target_layer = last_conv_layer
            self.layer_name = last_conv_layer.name
            print(f"✅ Auto-detected last conv layer: {self.layer_name}")
        else:
            # Fallback to specified layer_name
            try:
                target_layer = self.model.get_layer(self.layer_name)
                print(f"✅ Using specified layer: {self.layer_name}")
            except ValueError:
                raise ValueError(
                    f"Could not find convolutional layer. "
                    f"Tried auto-detection and specified layer '{self.layer_name}'. "
                    f"Available layers: {[l.name for l in self.model.layers]}"
                )
        
        # Handle model inputs - could be a list or single tensor
        if isinstance(self.model.inputs, list):
            model_inputs = self.model.inputs
        else:
            model_inputs = [self.model.inputs]
        
        # Handle model outputs - could be a list or single tensor
        if isinstance(self.model.output, list):
            model_output = self.model.output[0]  # Take first output if multiple
        else:
            model_output = self.model.output
        
        score_model = tf.keras.Model(
            inputs=model_inputs,
            outputs=[target_layer.output, model_output]
        )
        
        return score_model
    
    def generate_scorecam(self, img_array: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Generate Score-CAM heatmap for the specified class.
        
        Score-CAM algorithm:
        1. Extract feature maps from last conv layer
        2. For each feature map channel:
           - Upsample feature map to input image size
           - Create masked input (original image * normalized feature map)
           - Forward pass to get confidence score for target class
           - Use confidence score as weight
        3. Weight feature maps by their scores
        4. Average across channels
        5. Apply ReLU and normalize
        
        Args:
            img_array: Preprocessed image array [1, 224, 224, 3]
            class_idx: Target class index for Score-CAM
        
        Returns:
            Score-CAM heatmap as numpy array [H, W] normalized to [0, 1]
        """
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img_array)
        
        # Get feature maps and predictions from score model
        outputs = self.score_model(img_tensor)
        # Handle both single output and multiple outputs
        if isinstance(outputs, (list, tuple)):
            conv_outputs, predictions = outputs[0], outputs[1]
        else:
            # If single output, assume it's the predictions (fallback)
            conv_outputs = outputs
            predictions = outputs
        
        # Handle predictions shape - could be [batch, classes] or [classes]
        if len(predictions.shape) == 1:
            # 1D array: [classes]
            base_score = float(predictions[class_idx].numpy())
        elif len(predictions.shape) == 2:
            # 2D array: [batch, classes]
            base_score = float(predictions[0, class_idx].numpy())
        else:
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
        
        # Remove batch dimension from conv outputs
        conv_outputs = conv_outputs[0]  # Shape: [H, W, C]
        conv_outputs_np = conv_outputs.numpy()
        
        # Get feature map dimensions
        h, w, c = conv_outputs_np.shape
        img_h, img_w = img_array.shape[1], img_array.shape[2]  # 224, 224
        
        # Store scores for each feature map channel
        channel_scores = np.zeros(c)
        
        # For each feature map channel, compute forward pass score
        print(f"📊 Computing Score-CAM: processing {c} feature map channels...")
        for i in range(c):
            # Extract single channel feature map
            feature_map = conv_outputs_np[:, :, i]  # Shape: [H, W]
            
            # Normalize feature map to [0, 1]
            if feature_map.max() > feature_map.min():
                feature_map_norm = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            else:
                feature_map_norm = feature_map
            
            # Upsample feature map to input image size
            feature_map_upsampled = cv2.resize(
                feature_map_norm,
                (img_w, img_h),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Expand dimensions for broadcasting: [H, W] -> [1, H, W, 1]
            feature_map_upsampled = np.expand_dims(feature_map_upsampled, axis=(0, -1))
            
            # Create masked input: original image * normalized feature map
            # img_array shape: [1, H, W, 3], feature_map_upsampled: [1, H, W, 1]
            masked_input = img_array * feature_map_upsampled
            
            # Forward pass to get confidence score for target class
            masked_tensor = tf.convert_to_tensor(masked_input, dtype=tf.float32)
            masked_predictions = self.model(masked_tensor)
            
            # Get confidence score for target class
            if len(masked_predictions.shape) == 1:
                score = float(masked_predictions[class_idx].numpy())
            elif len(masked_predictions.shape) == 2:
                score = float(masked_predictions[0, class_idx].numpy())
            else:
                score = 0.0
            
            # Store score (subtract base score to get relative importance)
            channel_scores[i] = max(0, score - base_score)  # ReLU-like: only positive contributions
        
        # Weight each feature map by its score
        weighted_feature_maps = np.zeros((h, w))
        for i in range(c):
            weighted_feature_maps += conv_outputs_np[:, :, i] * channel_scores[i]
        
        # Average across all feature maps (already weighted)
        heatmap = weighted_feature_maps
        
        # Apply ReLU to focus on positive influences
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        else:
            # If all zeros, return uniform heatmap
            heatmap = np.ones_like(heatmap) * 0.5
        
        return heatmap
    
    def apply_jet_colormap(self, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Apply Jet colormap to heatmap.
        
        Args:
            heatmap: Grayscale heatmap [H, W] in range [0, 1]
            alpha: Opacity of the heatmap overlay
        
        Returns:
            Colored heatmap [H, W, 3] in range [0, 255]
        """
        # Convert to uint8 for colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        
        # Apply Jet colormap
        jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
        
        # Apply alpha
        jet_heatmap = jet_heatmap * alpha
        
        return jet_heatmap
    
    def overlay_heatmap_on_image(
        self, 
        original_image: Image.Image, 
        heatmap: np.ndarray,
        alpha: float = 0.6,
        colormap_alpha: float = 0.6
    ) -> Image.Image:
        """
        Overlay Score-CAM heatmap on the original image.
        
        Args:
            original_image: Original PIL Image
            heatmap: Score-CAM heatmap [H, W] in range [0, 1]
            alpha: Overall opacity of the overlay
            colormap_alpha: Opacity of the colormap itself
        
        Returns:
            PIL Image with heatmap overlay
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(
            heatmap, 
            (original_image.width, original_image.height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply colormap
        colored_heatmap = self.apply_jet_colormap(heatmap_resized, colormap_alpha)
        
        # Convert original image to array
        img_array = np.array(original_image, dtype=np.float32)
        
        # Blend images
        superimposed_img = img_array * (1 - alpha) + colored_heatmap * alpha
        
        # Clip and convert to uint8
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(superimposed_img)


def generate_scorecam_for_image(
    model,
    img_array: np.ndarray,
    original_image: Image.Image,
    class_idx: int,
    layer_name: str = None  # Auto-detect if None
) -> dict:
    """
    Convenience function to generate complete Score-CAM visualization.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array [1, 224, 224, 3]
        original_image: Original PIL Image
        class_idx: Target class index
        layer_name: Convolutional layer name
    
    Returns:
        Dictionary with:
            - 'heatmap': Raw heatmap array
            - 'colored_heatmap': Colored heatmap image
            - 'overlay': Heatmap overlaid on original image
    """
    # Create Score-CAM generator (auto-detects layer if layer_name is None)
    scorecam = ScoreCAMGenerator(model, layer_name or 'auto')
    
    # Generate heatmap
    heatmap = scorecam.generate_scorecam(img_array, class_idx)
    
    # Create colored heatmap
    colored_heatmap = scorecam.apply_jet_colormap(heatmap)
    colored_heatmap_img = Image.fromarray(colored_heatmap.astype(np.uint8))
    
    # Overlay on original image
    overlay_img = scorecam.overlay_heatmap_on_image(original_image, heatmap)
    
    return {
        'heatmap': heatmap,
        'colored_heatmap': colored_heatmap_img,
        'overlay': overlay_img
    }
