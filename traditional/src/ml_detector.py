"""
ESPMotion - ML Motion Detector

Neural network-based motion detector implementing the IDetector interface.

FOUNDATION: This is a working on-device inference path — an MLP (12 -> 16 -> 8 -> 1)
running on the pre-trained weights in ml_weights.py. The host-side training pipeline
that produced those weights is NOT part of this codebase; it was removed in the slim-down
and can be rebuilt from scratch. See docs/HOW_IT_WORKS.md ("ML detector (foundation)")
for the architecture, the 12 features, and how to regenerate ml_weights.py.

Enable it by setting DETECTION_ALGORITHM = "ml" in config.py.

Usage:
    from ml_detector import MLDetector

    detector = MLDetector()
    detector.process_packet(csi_data, subcarriers)
    metrics = detector.update_state()

"""
import math

try:
    from src.detector_interface import IDetector, MotionState
    from src.segmentation import SegmentationContext
    from src.features import extract_features_by_name, DEFAULT_FEATURES
    from src.config import DEFAULT_SUBCARRIERS
    from src.ml_weights import (
        FEATURE_MEAN, FEATURE_SCALE,
        W1, B1, W2, B2, W3, B3
    )
except ImportError:
    from detector_interface import IDetector, MotionState
    from segmentation import SegmentationContext
    from features import extract_features_by_name, DEFAULT_FEATURES
    from config import DEFAULT_SUBCARRIERS
    from ml_weights import (
        FEATURE_MEAN, FEATURE_SCALE,
        W1, B1, W2, B2, W3, B3
    )

# Re-export for convenience
__all__ = ['MLDetector', 'predict', 'is_motion', 'DEFAULT_SUBCARRIERS']

# ML-specific constants (unified with MVS for consistent UI)
ML_DEFAULT_THRESHOLD = 5.0
ML_MIN_THRESHOLD = 0.0
ML_MAX_THRESHOLD = 10.0
ML_METRIC_SCALE = 10.0

# Pre-allocated scratch for predict() — sizes match the MLP architecture
# (12 inputs, 16 hidden, 8 hidden). These are reused on every inference so
# the forward pass allocates nothing.
_NORMALIZED_SCRATCH = [0.0] * 12
_H1_SCRATCH = [0.0] * 16
_H2_SCRATCH = [0.0] * 8

# ============================================================================
# Neural Network Inference Functions
# ============================================================================

def relu(x):
    """ReLU activation function."""
    return x if x > 0 else 0.0


def sigmoid(x):
    """Sigmoid activation function with overflow protection."""
    if x < -20:
        return 0.0
    if x > 20:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def normalize_features(features):
    """Normalize features in-place into the shared scratch buffer."""
    x = _NORMALIZED_SCRATCH
    for i in range(len(features)):
        x[i] = (features[i] - FEATURE_MEAN[i]) / FEATURE_SCALE[i]
    return x


def predict(features):
    """
    Predict motion probability from 12 features.

    Architecture: 12 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)

    Reuses module-level scratch buffers — zero allocations per call.

    Args:
        features: List of 12 feature values

    Returns:
        float: Scaled motion metric (0.0 to 10.0)
    """
    # Normalize (writes into _NORMALIZED_SCRATCH)
    x = normalize_features(features)
    h1 = _H1_SCRATCH
    h2 = _H2_SCRATCH

    # Layer 1: 12 -> 16 (ReLU)
    for j in range(16):
        val = B1[j]
        for i in range(12):
            val += x[i] * W1[i][j]
        h1[j] = val if val > 0 else 0.0

    # Layer 2: 16 -> 8 (ReLU)
    for j in range(8):
        val = B2[j]
        for i in range(16):
            val += h1[i] * W2[i][j]
        h2[j] = val if val > 0 else 0.0

    # Layer 3: 8 -> 1 (Sigmoid)
    out = B3[0]
    for i in range(8):
        out += h2[i] * W3[i][0]

    return sigmoid(out) * ML_METRIC_SCALE


def is_motion(features, threshold=ML_DEFAULT_THRESHOLD):
    """
    Detect motion from features.
    
    Args:
        features: List of 12 feature values
        threshold: Detection threshold (default: 5.0)
    
    Returns:
        bool: True if motion detected
    """
    return predict(features) > threshold


# ============================================================================
# MLDetector Class
# ============================================================================


class MLDetector(IDetector):
    """
    Neural Network-based motion detector.
    
    Uses a pre-trained MLP (12 -> 16 -> 8 -> 1) to classify
    motion based on turbulence features extracted from CSI data.
    
    Algorithm:
    1. Calculate spatial turbulence (std of subcarrier amplitudes)
    2. Store in circular buffer (window_size packets)
    3. Extract 12 statistical features from buffer
    4. Run neural network inference
    5. Compare probability to threshold for state decision
    """
    
    def __init__(self, window_size=75, threshold=ML_DEFAULT_THRESHOLD,
                 enable_lowpass=False, lowpass_cutoff=11.0,
                 enable_hampel=True, hampel_window=7, hampel_threshold=5.0,
                 use_cv_normalization=False):
        """
        Initialize ML detector.
        
        Args:
            window_size: Feature extraction window size (default: 75, matches C++ DETECTOR_DEFAULT_WINDOW_SIZE)
            threshold: Motion detection threshold (default: 5.0, range 0.0-10.0)
            enable_lowpass: Enable low-pass filter (default: False)
            lowpass_cutoff: Low-pass cutoff frequency Hz (default: 11.0)
            enable_hampel: Enable Hampel filter (default: True, model trained with Hampel)
            hampel_window: Hampel window size (default: 7)
            hampel_threshold: Hampel threshold in MAD (default: 5.0)
            use_cv_normalization: Use CV (std/mean) instead of raw std (default: False)
                                  Set True for chips without gain lock (e.g., ESP32)
        """
        # Use SegmentationContext for turbulence calculation and filtering
        self._context = SegmentationContext(
            window_size=window_size,
            threshold=1.0,  # Not used, we use probability threshold
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold
        )
        # CV normalization: True for chips without gain lock (ESP32)
        # False for chips with gain lock (C3, C6, S3) - raw std is more sensitive
        self._context.use_cv_normalization = use_cv_normalization
        self._threshold = threshold
        self._packet_count = 0
        self._motion_count = 0
        self._state = MotionState.IDLE
        self._current_probability = 0.0
        
        # For tracking (optional)
        self.probability_history = []
        self.state_history = []
        self.track_data = False

        # Store current amplitudes for feature extraction
        self._current_amplitudes = None

        # Pre-allocated chronological mirror of the circular turbulence
        # buffer. Filled in-place each publish so _extract_features() never
        # allocates a slice+concat (the old hot path: ~1.8 KB per publish).
        self._chrono_buf = [0.0] * window_size

        # Reusable metrics dict — mutated in place each update_state() so
        # publish never allocates a fresh dict.
        self._metrics = {
            'state': self._state,
            'probability': 0.0,
            'threshold': self._threshold,
        }
    
    def process_packet(self, csi_data, selected_subcarriers=None):
        """
        Process a CSI packet.
        
        Args:
            csi_data: Raw CSI data (int8 I/Q pairs)
            selected_subcarriers: Subcarrier indices to use
        """
        self._packet_count += 1
        
        # Calculate spatial turbulence using instance method (CV-normalized)
        # Also get amplitudes for cross-subcarrier features
        turbulence, amplitudes = self._context.calculate_spatial_turbulence(
            csi_data, selected_subcarriers, return_amplitudes=True
        )
        
        # Store amplitudes for feature extraction
        self._current_amplitudes = amplitudes
        
        # Add to buffer
        self._context.add_turbulence(turbulence)
    
    def update_state(self):
        """
        Run inference and update state.

        Returns:
            dict: Current metrics (the same instance dict is mutated and
            returned each call — callers must read values immediately).
        """
        m = self._metrics
        if not self.is_ready():
            m['state'] = self._state
            m['probability'] = 0.0
            m['threshold'] = self._threshold
            return m

        # Extract features from turbulence buffer
        features = self._extract_features()

        # Run neural network
        self._current_probability = predict(features)

        # Update state
        if self._current_probability > self._threshold:
            self._state = MotionState.MOTION
        else:
            self._state = MotionState.IDLE

        if self.track_data:
            self.probability_history.append(self._current_probability)
            state_str = 'MOTION' if self._state == MotionState.MOTION else 'IDLE'
            self.state_history.append(state_str)
            if self._state == MotionState.MOTION:
                self._motion_count += 1

        m['state'] = self._state
        m['probability'] = self._current_probability
        m['threshold'] = self._threshold
        return m
    
    def _extract_features(self):
        """
        Extract 12 features from turbulence buffer using centralized extractor.

        The turbulence_buffer is a circular buffer; after wrap-around a slice
        is not chronological, and features like slope/zcr/autocorr depend on
        temporal order. We copy into a pre-allocated chronological mirror
        in place — no slice, no concat, zero allocations.
        """
        ctx = self._context
        src = ctx.turbulence_buffer
        dst = self._chrono_buf

        if ctx.buffer_count < ctx.window_size:
            # Buffer not full yet: data is in order from index 0
            n = ctx.buffer_count
            for i in range(n):
                dst[i] = src[i]
        else:
            # Wrapped: copy [idx:] then [:idx] into dst.
            idx = ctx.buffer_index
            ws = ctx.window_size
            n = ws
            j = 0
            for i in range(idx, ws):
                dst[j] = src[i]
                j += 1
            for i in range(0, idx):
                dst[j] = src[i]
                j += 1

        return extract_features_by_name(
            dst, n,
            amplitudes=self._current_amplitudes,
            feature_names=DEFAULT_FEATURES
        )
    
    def get_state(self):
        """Get current motion state."""
        return self._state
    
    def get_motion_metric(self):
        """Get current motion probability."""
        return self._current_probability
    
    def get_threshold(self):
        """Get current probability threshold."""
        return self._threshold
    
    def set_threshold(self, threshold):
        """Set threshold (range 0.0-10.0, unified with MVS)."""
        if ML_MIN_THRESHOLD <= threshold <= ML_MAX_THRESHOLD:
            self._threshold = threshold
            return True
        return False
    
    def is_ready(self):
        """Check if buffer is full."""
        return self._context.buffer_count >= self._context.window_size
    
    def reset(self):
        """Reset detector state."""
        self._context.reset(full=True)
        self._state = MotionState.IDLE
        self._current_probability = 0.0
        self._motion_count = 0
        self.probability_history = []
        self.state_history = []
    
    def get_name(self):
        """Get detector name."""
        return "ML"
    
    @property
    def total_packets(self):
        """Total packets processed."""
        return self._packet_count
    
    def get_motion_count(self):
        """Get number of motion detections (for tracking)."""
        return self._motion_count
