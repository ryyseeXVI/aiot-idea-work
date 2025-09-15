"""
Embedded AI Model Converter
Creates ultra-lightweight models for microcontroller deployment
"""

import torch
import torch.nn as nn
import numpy as np
import json
import struct
from typing import List, Dict, Any
from pathlib import Path

class MicroControllerAI:
    """
    Ultra-lightweight AI implementation for microcontrollers
    Uses quantized weights and minimal computation
    """
    
    def __init__(self):
        # Quantized neural network weights (INT8)
        self.weights = {}
        self.biases = {}
        self.layer_sizes = [6, 16, 8, 1]  # Input -> Hidden1 -> Hidden2 -> Output
        self.feature_means = None
        self.feature_stds = None
        
    def load_from_pytorch(self, model_path: str, scaler_path: str):
        """Convert PyTorch model to microcontroller format"""
        try:
            import joblib
            
            # Load PyTorch model
            model = torch.load(model_path, map_location='cpu')
            
            # Load scaler
            scaler = joblib.load(scaler_path)
            self.feature_means = scaler.mean_.astype(np.float32)
            self.feature_stds = scaler.scale_.astype(np.float32)
            
            # Extract and quantize weights
            layer_idx = 0
            for name, param in model.items():
                if 'weight' in name:
                    # Quantize weights to INT8 (-128 to 127)
                    weight_data = param.numpy().astype(np.float32)
                    scale = np.max(np.abs(weight_data)) / 127.0
                    quantized = np.round(weight_data / scale).astype(np.int8)
                    
                    self.weights[f'layer_{layer_idx}'] = {
                        'data': quantized,
                        'scale': scale
                    }
                elif 'bias' in name:
                    # Keep biases as float32 for accuracy
                    self.biases[f'layer_{layer_idx}'] = param.numpy().astype(np.float32)
                    layer_idx += 1
            
            print(f"✅ Converted model: {len(self.weights)} layers")
            print(f"   Model size: ~{self._calculate_model_size()} bytes")
            return True
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False
    
    def _calculate_model_size(self) -> int:
        """Calculate total model size in bytes"""
        size = 0
        for layer_data in self.weights.values():
            size += layer_data['data'].nbytes  # INT8 weights
            size += 4  # Float32 scale
        for bias_data in self.biases.values():
            size += bias_data.nbytes  # Float32 biases
        size += len(self.feature_means) * 8  # Mean + std for each feature
        return size
    
    def predict(self, features: List[float]) -> tuple[bool, float]:
        """
        Perform inference using quantized model
        Optimized for microcontroller execution
        """
        # Normalize features (equivalent to sklearn StandardScaler)
        normalized = [(f - m) / s for f, m, s in zip(features, self.feature_means, self.feature_stds)]
        
        # Forward pass through quantized network
        activations = np.array(normalized, dtype=np.float32)
        
        for i in range(len(self.layer_sizes) - 1):
            layer_key = f'layer_{i}'
            
            # Quantized matrix multiplication
            weights = self.weights[layer_key]
            quantized_weights = weights['data'].astype(np.float32) * weights['scale']
            biases = self.biases[layer_key]
            
            # Linear transformation
            activations = np.dot(activations, quantized_weights.T) + biases
            
            # Activation function (ReLU for hidden layers, Sigmoid for output)
            if i < len(self.layer_sizes) - 2:  # Hidden layers
                activations = np.maximum(0, activations)  # ReLU
            else:  # Output layer
                activations = 1.0 / (1.0 + np.exp(-np.clip(activations, -500, 500)))  # Sigmoid
        
        confidence = float(activations[0])
        decision = confidence > 0.5
        
        return decision, confidence
    
    def export_c_header(self, output_path: str = "ai_model.h"):
        """Export model as C header file for embedded systems"""
        
        with open(output_path, 'w') as f:
            f.write("/* Auto-generated AI model for microcontroller */\n")
            f.write("#ifndef AI_MODEL_H\n#define AI_MODEL_H\n\n")
            f.write("#include <stdint.h>\n\n")
            
            # Feature normalization constants
            f.write("/* Feature normalization */\n")
            f.write(f"#define NUM_FEATURES {len(self.feature_means)}\n")
            f.write("static const float FEATURE_MEANS[] = {")
            f.write(", ".join([f"{m:.6f}f" for m in self.feature_means]))
            f.write("};\n")
            f.write("static const float FEATURE_STDS[] = {")
            f.write(", ".join([f"{s:.6f}f" for s in self.feature_stds]))
            f.write("};\n\n")
            
            # Network architecture
            f.write("/* Network architecture */\n")
            f.write(f"#define NUM_LAYERS {len(self.layer_sizes) - 1}\n")
            f.write("static const int LAYER_SIZES[] = {")
            f.write(", ".join(map(str, self.layer_sizes)))
            f.write("};\n\n")
            
            # Quantized weights
            for i, (layer_key, weight_data) in enumerate(self.weights.items()):
                weights = weight_data['data']
                scale = weight_data['scale']
                
                f.write(f"/* Layer {i} weights (quantized INT8) */\n")
                f.write(f"static const int8_t WEIGHTS_L{i}[] = {{\n")
                
                # Write weights in rows
                for row in range(weights.shape[0]):
                    f.write("  ")
                    for col in range(weights.shape[1]):
                        f.write(f"{weights[row, col]:4d}")
                        if col < weights.shape[1] - 1:
                            f.write(", ")
                    if row < weights.shape[0] - 1:
                        f.write(",")
                    f.write("\n")
                f.write("};\n")
                f.write(f"static const float SCALE_L{i} = {scale:.8f}f;\n\n")
                
                # Biases
                biases = self.biases[layer_key]
                f.write(f"static const float BIASES_L{i}[] = {{")
                f.write(", ".join([f"{b:.6f}f" for b in biases]))
                f.write("};\n\n")
            
            # Inference function
            f.write(self._generate_c_inference_function())
            f.write("\n#endif /* AI_MODEL_H */\n")
        
        print(f"✅ C header exported to {output_path}")
    
    def _generate_c_inference_function(self) -> str:
        """Generate optimized C inference function"""
        return """/* Optimized inference function for microcontroller */
static inline float relu(float x) {
    return x > 0 ? x : 0;
}

static inline float sigmoid(float x) {
    if (x > 500) return 1.0f;
    if (x < -500) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* Main AI inference function */
int ai_predict(const float* features, float* confidence) {
    float normalized[NUM_FEATURES];
    float layer_output[16];  // Max layer size
    
    // Normalize input features
    for (int i = 0; i < NUM_FEATURES; i++) {
        normalized[i] = (features[i] - FEATURE_MEANS[i]) / FEATURE_STDS[i];
    }
    
    // Copy input to layer output
    for (int i = 0; i < NUM_FEATURES; i++) {
        layer_output[i] = normalized[i];
    }
    
    // Forward pass through layers
    float temp_output[16];
    
    // Layer 0: 6 -> 16
    for (int out = 0; out < 16; out++) {
        float sum = BIASES_L0[out];
        for (int in = 0; in < 6; in++) {
            int8_t weight = WEIGHTS_L0[out * 6 + in];
            sum += layer_output[in] * (weight * SCALE_L0);
        }
        temp_output[out] = relu(sum);
    }
    for (int i = 0; i < 16; i++) layer_output[i] = temp_output[i];
    
    // Layer 1: 16 -> 8
    for (int out = 0; out < 8; out++) {
        float sum = BIASES_L1[out];
        for (int in = 0; in < 16; in++) {
            int8_t weight = WEIGHTS_L1[out * 16 + in];
            sum += layer_output[in] * (weight * SCALE_L1);
        }
        temp_output[out] = relu(sum);
    }
    for (int i = 0; i < 8; i++) layer_output[i] = temp_output[i];
    
    // Layer 2: 8 -> 1 (output)
    float sum = BIASES_L2[0];
    for (int in = 0; in < 8; in++) {
        int8_t weight = WEIGHTS_L2[in];
        sum += layer_output[in] * (weight * SCALE_L2);
    }
    float output = sigmoid(sum);
    
    *confidence = output;
    return output > 0.5f ? 1 : 0;
}"""

    def export_micropython(self, output_path: str = "ai_model.py"):
        """Export model as MicroPython module"""
        
        with open(output_path, 'w') as f:
            f.write('"""Auto-generated AI model for MicroPython"""\n')
            f.write("import math\n\n")
            
            # Feature normalization
            f.write("# Feature normalization constants\n")
            f.write(f"FEATURE_MEANS = {list(self.feature_means)}\n")
            f.write(f"FEATURE_STDS = {list(self.feature_stds)}\n\n")
            
            # Model weights and biases
            f.write("# Quantized model weights and biases\n")
            f.write("MODEL_DATA = {\n")
            
            for i, (layer_key, weight_data) in enumerate(self.weights.items()):
                weights = weight_data['data'].tolist()
                scale = weight_data['scale']
                biases = self.biases[layer_key].tolist()
                
                f.write(f"    'layer_{i}': {{\n")
                f.write(f"        'weights': {weights},\n")
                f.write(f"        'scale': {scale},\n")
                f.write(f"        'biases': {biases}\n")
                f.write("    },\n")
            
            f.write("}\n\n")
            
            # Inference functions
            f.write("""def relu(x):
    return max(0, x)

def sigmoid(x):
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def ai_predict(features):
    '''Perform AI inference on microcontroller'''
    # Normalize features
    normalized = [(f - m) / s for f, m, s in zip(features, FEATURE_MEANS, FEATURE_STDS)]
    
    # Forward pass
    activations = normalized[:]
    
    for layer_idx in range(len(MODEL_DATA)):
        layer_data = MODEL_DATA[f'layer_{layer_idx}']
        weights = layer_data['weights']
        scale = layer_data['scale']
        biases = layer_data['biases']
        
        # Matrix multiplication with quantized weights
        new_activations = []
        for out_idx in range(len(biases)):
            sum_val = biases[out_idx]
            for in_idx in range(len(activations)):
                weight = weights[out_idx][in_idx] * scale
                sum_val += activations[in_idx] * weight
            
            # Apply activation function
            if layer_idx < len(MODEL_DATA) - 1:  # Hidden layers
                new_activations.append(relu(sum_val))
            else:  # Output layer
                new_activations.append(sigmoid(sum_val))
        
        activations = new_activations
    
    confidence = activations[0]
    decision = confidence > 0.5
    return decision, confidence

# Example usage:
# decision, confidence = ai_predict([25.5, 65.0, 600, 0, 1, 0])
# print(f"Relay: {'ON' if decision else 'OFF'}, Confidence: {confidence:.3f}")
""")
        
        print(f"✅ MicroPython module exported to {output_path}")

def convert_model_for_embedded():
    """Convert trained PyTorch model for embedded deployment"""
    print("🔧 Converting AI model for embedded deployment...")
    
    # Initialize converter
    converter = MicroControllerAI()
    
    # Convert PyTorch model
    model_path = "../model/relay_model.pth"
    scaler_path = "../model/scaler.pkl"
    
    if converter.load_from_pytorch(model_path, scaler_path):
        # Export for different platforms
        converter.export_c_header("embedded_ai_model.h")
        converter.export_micropython("embedded_ai_model.py")
        
        # Test the converted model
        test_features = [25.5, 65.0, 600, 0, 1, 0]  # temp, humidity, vibration, leak, off_peak, alert
        decision, confidence = converter.predict(test_features)
        print(f"🧪 Test prediction: Relay {'ON' if decision else 'OFF'} (confidence: {confidence:.3f})")
        
        return True
    else:
        print("❌ Model conversion failed")
        return False

if __name__ == "__main__":
    convert_model_for_embedded()
