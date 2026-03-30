const std = @import("std");
const cortex = @import("cortex");
const Tensor = cortex.Tensor;
const Dense = cortex.Dense;
const ReLU = cortex.ReLU;
const Sigmoid = cortex.Sigmoid;
const Sequential = cortex.Sequential;
const MSELoss = cortex.MSELoss;
const SGD = cortex.SGD;

// Train a network on XOR problem
// XOR:
//   [0, 0] -> 0
//   [0, 1] -> 1
//   [1, 0] -> 1
//   [1, 1] -> 0
test "train xor" {
    const allocator = std.testing.allocator;

    // Create XOR dataset
    var inputs = try Tensor.init(allocator, &[_]usize{ 4, 2 });
    defer inputs.deinit();
    // [0, 0]
    inputs.data[0] = 0.0;
    inputs.data[1] = 0.0;
    // [0, 1]
    inputs.data[2] = 0.0;
    inputs.data[3] = 1.0;
    // [1, 0]
    inputs.data[4] = 1.0;
    inputs.data[5] = 0.0;
    // [1, 1]
    inputs.data[6] = 1.0;
    inputs.data[7] = 1.0;

    var targets = try Tensor.init(allocator, &[_]usize{ 4, 1 });
    defer targets.deinit();
    targets.data[0] = 0.0;
    targets.data[1] = 1.0;
    targets.data[2] = 1.0;
    targets.data[3] = 0.0;

    // Build network: 2 -> 8 -> 1
    var dense1 = try Dense.init(allocator, 2, 8);
    defer dense1.deinit();

    var relu = ReLU.init(allocator);
    defer relu.deinit();

    var dense2 = try Dense.init(allocator, 8, 1);
    defer dense2.deinit();

    var sigmoid = Sigmoid.init(allocator);
    defer sigmoid.deinit();

    // Initialize weights with better values
    var rng = cortex.Random.init(42);
    rng.fillUniform(&dense1.weights, -1.0, 1.0);
    rng.fillUniform(&dense2.weights, -1.0, 1.0);

    // Build model
    var model = Sequential.init(allocator);
    defer model.deinit();
    try model.addDense(&dense1);
    try model.addReLU(&relu);
    try model.addDense(&dense2);
    try model.addSigmoid(&sigmoid);

    // Loss function
    var loss_fn = MSELoss.init(allocator);
    defer loss_fn.deinit();

    // Optimizer
    var optimizer = SGD.init(allocator, 1.0, 0.9);
    defer optimizer.deinit();
    try optimizer.addParam(&dense1.weights);
    try optimizer.addParam(&dense1.bias);
    try optimizer.addParam(&dense2.weights);
    try optimizer.addParam(&dense2.bias);

    // Training loop
    var final_loss: f32 = 1.0;
    for (0..1000) |_| {
        // Forward pass
        var output = try model.forward(&inputs);
        defer output.deinit();

        // Compute loss
        final_loss = try loss_fn.forward(&output, &targets);

        // Backward pass - manual gradient computation for this simple case
        var grad_output = try loss_fn.backward();
        defer grad_output.deinit();

        // Backward through sigmoid
        var grad_sigmoid = try sigmoid.backward(&grad_output);
        defer grad_sigmoid.deinit();

        // Allocate gradient tensors for dense2
        var grad_w2 = try Tensor.zeros(allocator, dense2.weights.shape);
        defer grad_w2.deinit();
        var grad_b2 = try Tensor.zeros(allocator, dense2.bias.shape);
        defer grad_b2.deinit();

        // Backward through dense2
        var grad_relu = try dense2.backward(&grad_sigmoid, &grad_w2, &grad_b2);
        defer grad_relu.deinit();

        // Backward through relu
        var grad_relu_in = try relu.backward(&grad_relu);
        defer grad_relu_in.deinit();

        // Allocate gradient tensors for dense1
        var grad_w1 = try Tensor.zeros(allocator, dense1.weights.shape);
        defer grad_w1.deinit();
        var grad_b1 = try Tensor.zeros(allocator, dense1.bias.shape);
        defer grad_b1.deinit();

        // Backward through dense1
        var grad_input = try dense1.backward(&grad_relu_in, &grad_w1, &grad_b1);
        defer grad_input.deinit();

        // Update weights (SGD modifies tensors in-place via data slices)
        // The tensors share data with dense1/dense2, so modifications persist
        var params = [_]Tensor{ dense1.weights, dense1.bias, dense2.weights, dense2.bias };
        const grads = [_]Tensor{ grad_w1, grad_b1, grad_w2, grad_b2 };
        try optimizer.step(&params, &grads);

        if (final_loss < 0.01) break;
    }

    // Check that loss is low
    try std.testing.expect(final_loss < 0.1);

    // Test predictions
    var predictions = try model.forward(&inputs);
    defer predictions.deinit();

    // XOR outputs should be close to targets
    // Allow some tolerance
    try std.testing.expect(predictions.data[0] < 0.3); // [0,0] -> ~0
    try std.testing.expect(predictions.data[1] > 0.7); // [0,1] -> ~1
    try std.testing.expect(predictions.data[2] > 0.7); // [1,0] -> ~1
    try std.testing.expect(predictions.data[3] < 0.3); // [1,1] -> ~0
}
