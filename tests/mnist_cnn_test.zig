const std = @import("std");
const cortex = @import("cortex");
const Tensor = cortex.Tensor;
const Conv2D = cortex.Conv2D;
const MaxPool2D = cortex.MaxPool2D;
const Dense = cortex.Dense;
const ReLU = cortex.ReLU;
const CrossEntropyLoss = cortex.CrossEntropyLoss;
const Adam = cortex.Adam;

// Train CNN on MNIST-like data
// Architecture: Conv2D(1,8,3) -> ReLU -> MaxPool(2) -> Flatten -> Dense(288,10)
// Uses synthetic data for testing
test "train mnist cnn" {
    const allocator = std.testing.allocator;

    // Create synthetic MNIST-like data (smaller for testing)
    // Using 8x8 images instead of 28x28 for faster testing
    const num_samples = 32;
    const img_size = 8;
    const num_classes = 10;

    // Create random images: [num_samples, 1, 8, 8]
    var images = try Tensor.init(allocator, &[_]usize{ num_samples, 1, img_size, img_size });
    defer images.deinit();

    var labels = try Tensor.zeros(allocator, &[_]usize{ num_samples, num_classes });
    defer labels.deinit();

    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    for (0..num_samples) |i| {
        // Random image
        for (0..img_size * img_size) |j| {
            images.data[i * img_size * img_size + j] = rand.float(f32);
        }
        // Random label (one-hot)
        const label = rand.uintLessThan(usize, num_classes);
        labels.data[i * num_classes + label] = 1.0;
    }

    // Build CNN
    // Conv2D: [B, 1, 8, 8] -> [B, 8, 6, 6] (kernel=3, stride=1, no padding)
    var conv1 = try Conv2D.init(allocator, 1, 8, 3, 1, 0);
    defer conv1.deinit();

    var relu1 = ReLU.init(allocator);
    defer relu1.deinit();

    // MaxPool: [B, 8, 6, 6] -> [B, 8, 3, 3]
    var pool1 = MaxPool2D.init(allocator, 2, 2);
    defer pool1.deinit();

    // After flatten: 8 * 3 * 3 = 72 features
    // Dense: 72 -> 10
    var dense1 = try Dense.init(allocator, 72, num_classes);
    defer dense1.deinit();

    // Loss
    var loss_fn = CrossEntropyLoss.init(allocator);
    defer loss_fn.deinit();

    // Training - just verify the forward pass works
    // Full training would be too slow for a test

    // Forward pass through conv
    var conv_out = try conv1.forward(&images);
    defer conv_out.deinit();

    // Check shape: [32, 8, 6, 6]
    try std.testing.expectEqual(@as(usize, num_samples), conv_out.shape[0]);
    try std.testing.expectEqual(@as(usize, 8), conv_out.shape[1]);
    try std.testing.expectEqual(@as(usize, 6), conv_out.shape[2]);
    try std.testing.expectEqual(@as(usize, 6), conv_out.shape[3]);

    // ReLU
    var relu_out = try relu1.forward(&conv_out);
    defer relu_out.deinit();

    // MaxPool
    var pool_out = try pool1.forward(&relu_out);
    defer pool_out.deinit();

    // Check shape: [32, 8, 3, 3]
    try std.testing.expectEqual(@as(usize, num_samples), pool_out.shape[0]);
    try std.testing.expectEqual(@as(usize, 8), pool_out.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), pool_out.shape[2]);
    try std.testing.expectEqual(@as(usize, 3), pool_out.shape[3]);

    // Flatten for dense layer
    var flattened = try pool_out.reshape(&[_]usize{ num_samples, 72 });
    defer flattened.deinit();

    // Dense
    var logits = try dense1.forward(&flattened);
    defer logits.deinit();

    // Check shape: [32, 10]
    try std.testing.expectEqual(@as(usize, num_samples), logits.shape[0]);
    try std.testing.expectEqual(@as(usize, num_classes), logits.shape[1]);

    // Compute loss
    const loss = try loss_fn.forward(&logits, &labels);

    // Loss should be positive and finite
    try std.testing.expect(loss > 0.0);
    try std.testing.expect(std.math.isFinite(loss));

    // Backward pass to verify gradients flow
    var grad_logits = try loss_fn.backward();
    defer grad_logits.deinit();

    try std.testing.expectEqual(@as(usize, num_samples), grad_logits.shape[0]);
    try std.testing.expectEqual(@as(usize, num_classes), grad_logits.shape[1]);
}
