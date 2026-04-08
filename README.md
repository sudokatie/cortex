# Cortex

Neural network library from scratch in Zig. No autograd magic - every gradient is computed explicitly.

## Why?

Because sometimes you want to see every matrix multiply. Every backpropagation step. Every weight update. This isn't for production ML (use PyTorch for that). This is for understanding how neural networks actually work.

## Features

- **Tensors** - Multi-dimensional arrays with broadcasting, matmul, reductions
- **Layers** - Dense, Conv2D, BatchNorm, Dropout, Attention
- **Activations** - ReLU, Sigmoid, Tanh, Softmax
- **Losses** - Cross-entropy, MSE
- **Optimizers** - SGD (with momentum), Adam
- **Models** - Sequential container for stacking layers
- **Serialization** - Save/load trained models to binary format
- **GPU Support** - Device abstraction with automatic CPU fallback

## Quick Start

```zig
const cortex = @import("cortex");

// Build model
var model = cortex.Sequential.init(allocator);
defer model.deinit();

var dense1 = try cortex.Dense.init(allocator, 784, 128);
defer dense1.deinit();
var relu = cortex.ReLU.init(allocator);
defer relu.deinit();
var dense2 = try cortex.Dense.init(allocator, 128, 10);
defer dense2.deinit();

try model.addDense(&dense1);
try model.addReLU(&relu);
try model.addDense(&dense2);

// Forward pass
var output = try model.forward(&input);
defer output.deinit();
```

## Model Serialization

Save and load trained models:

```zig
// Save model
try cortex.save(&model, "model.crtx");

// Load model
var loaded = try cortex.load(allocator, "model.crtx");
defer {
    for (loaded.layers.items) |layer| {
        switch (layer) {
            .dense => |d| { d.deinit(); allocator.destroy(d); },
            .relu => |r| { r.deinit(); allocator.destroy(r); },
            .sigmoid => |s| { s.deinit(); allocator.destroy(s); },
            .tanh => |t| { t.deinit(); allocator.destroy(t); },
        }
    }
    loaded.deinit();
}
```

### File Format

Binary format with magic bytes `CRTX`:
- Header: magic (4 bytes) + version (4 bytes) + layer count (4 bytes)
- Per layer: type (1 byte) + layer-specific data
- Dense layers store dimensions + weights + bias as raw f32

## Testing

```bash
zig build test
```

## GPU Support

The device module provides a backend abstraction for GPU acceleration:

```zig
const cortex = @import("cortex");

// Create tensor on specific device
var t = try cortex.DeviceTensor.initGpu(allocator, &[_]usize{1024, 1024}, 0);
defer t.deinit();

// Move tensor between devices
try t.toCpu();

// Operations fall back to CPU automatically when GPU unavailable
var a = try cortex.DeviceTensor.zeros(allocator, &[_]usize{4}, cortex.Device.cpu);
var b = try cortex.DeviceTensor.ones(allocator, &[_]usize{4}, cortex.Device.cpu);
var out = try cortex.DeviceTensor.init(allocator, &[_]usize{4}, cortex.Device.cpu);
try cortex.device_tensor.add(&a, &b, &out);
```

### Backend Architecture

- **Device** - CPU or GPU identifier with index
- **Backend** - Interface for memory allocation and operations
- **DeviceTensor** - GPU-aware tensor with automatic device management
- **Fallback** - Operations automatically fall back to CPU when GPU unavailable

GPU backends (Vulkan compute) are stubbed for future implementation. All operations work on CPU by default.

## Constraints

- Pure Zig (no external dependencies)
- f32 only (no mixed precision)
- No autograd - gradients are explicit

## License

MIT

---

Built by Katie.
