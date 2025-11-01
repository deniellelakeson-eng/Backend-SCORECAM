"""
HerbaScan Backend - Grad-CAM Implementation
True gradient-based Class Activation Mapping using TensorFlow
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


class GradCAMGenerator:
    """
    Generates Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations.
    
    This provides true gradient-based explanations of CNN predictions, showing
    which regions of the input image were most important for the classification.
    """
    
    def __init__(self, model, layer_name: str = 'Conv_1'):
        """
        Initialize Grad-CAM generator.
        
        Args:
            model: Trained Keras model
            layer_name: Name of the last convolutional layer to use for Grad-CAM
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = self._create_grad_model()
    
    def _create_grad_model(self):
        """
        Create gradient model that outputs both the target layer activations
        and the final predictions.
        """
        try:
            target_layer = self.model.get_layer(self.layer_name)
        except ValueError:
            # If layer not found, try to find the last conv layer automatically
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    target_layer = layer
                    self.layer_name = layer.name
                    print(f"Using layer: {self.layer_name}")
                    break
            else:
                raise ValueError(f"Could not find convolutional layer: {self.layer_name}")
        
        grad_model = tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[target_layer.output, self.model.output]
        )
        
        return grad_model
    
    def generate_gradcam(self, img_array: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the specified class.
        
        Args:
            img_array: Preprocessed image array [1, 224, 224, 3]
            class_idx: Target class index for Grad-CAM
        
        Returns:
            Grad-CAM heatmap as numpy array [H, W] normalized to [0, 1]
        """
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img_array)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            loss = predictions[:, class_idx]
        
        # Get gradients of the loss with respect to the conv layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the gradients
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Weight each feature map by its gradient
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average across all feature maps
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Apply ReLU to focus on positive influences
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
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
        Overlay Grad-CAM heatmap on the original image.
        
        Args:
            original_image: Original PIL Image
            heatmap: Grad-CAM heatmap [H, W] in range [0, 1]
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


def generate_gradcam_for_image(
    model,
    img_array: np.ndarray,
    original_image: Image.Image,
    class_idx: int,
    layer_name: str = 'Conv_1'
) -> dict:
    """
    Convenience function to generate complete Grad-CAM visualization.
    
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
    # Create Grad-CAM generator
    gradcam = GradCAMGenerator(model, layer_name)
    
    # Generate heatmap
    heatmap = gradcam.generate_gradcam(img_array, class_idx)
    
    # Create colored heatmap
    colored_heatmap = gradcam.apply_jet_colormap(heatmap)
    colored_heatmap_img = Image.fromarray(colored_heatmap.astype(np.uint8))
    
    # Overlay on original image
    overlay_img = gradcam.overlay_heatmap_on_image(original_image, heatmap)
    
    return {
        'heatmap': heatmap,
        'colored_heatmap': colored_heatmap_img,
        'overlay': overlay_img
    }

