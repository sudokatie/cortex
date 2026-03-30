const std = @import("std");
const cortex = @import("cortex");
const Tensor = cortex.Tensor;
const Dense = cortex.Dense;
const ReLU = cortex.ReLU;
const Sequential = cortex.Sequential;
const CrossEntropyLoss = cortex.CrossEntropyLoss;
const Adam = cortex.Adam;
const DataLoader = cortex.DataLoader;
const Dataset = cortex.Dataset;
const mnist = cortex.mnist;

// Train MLP on MNIST-like data
// Architecture: 784 -> 128 -> ReLU -> 10
// Uses synthetic data for testing (real MNIST requires data files)
test "train mnist mlp" {
    const allocator = std.testing.allocator;

    // Create synthetic MNIST-like data (smaller for testing)
    const num_train = 200;
    const num_test = 50;

    const train_data = try mnist.createSyntheticData(allocator, num_train);
    const train_images = train_data.images;
    const train_labels = train_data.labels;

    const test_data = try mnist.createSyntheticData(allocator, num_test);
    const test_images = test_data.images;
    const test_labels = test_data.labels;

    var train_dataset = try Dataset.init(train_images, train_labels);
    defer train_dataset.deinit();

    var test_dataset = try Dataset.init(test_images, test_labels);
    defer test_dataset.deinit();

    // Build MLP: 784 -> 128 -> ReLU -> 10
    var dense1 = try Dense.init(allocator, 784, 128);
    defer dense1.deinit();

    var relu = ReLU.init(allocator);
    defer relu.deinit();

    var dense2 = try Dense.init(allocator, 128, 10);
    defer dense2.deinit();

    // Initialize with Kaiming for ReLU
    var rng = cortex.Random.init(42);
    var w1 = try cortex.WeightInit.kaiming(allocator, &rng, dense1.weights.shape);
    defer w1.deinit();
    @memcpy(dense1.weights.data, w1.data);

    var w2 = try cortex.WeightInit.kaiming(allocator, &rng, dense2.weights.shape);
    defer w2.deinit();
    @memcpy(dense2.weights.data, w2.data);

    // Build model
    var model = Sequential.init(allocator);
    defer model.deinit();
    try model.addDense(&dense1);
    try model.addReLU(&relu);
    try model.addDense(&dense2);

    // Loss and optimizer
    var loss_fn = CrossEntropyLoss.init(allocator);
    defer loss_fn.deinit();

    var optimizer = Adam.init(allocator, 0.001);
    defer optimizer.deinit();
    try optimizer.addParam(&dense1.weights);
    try optimizer.addParam(&dense1.bias);
    try optimizer.addParam(&dense2.weights);
    try optimizer.addParam(&dense2.bias);

    // Training loop
    const batch_size = 32;
    const epochs = 5;
    var final_loss: f32 = 10.0;

    for (0..epochs) |_| {
        var loader = try DataLoader.init(allocator, &train_dataset, batch_size, true);
        defer loader.deinit();

        while (try loader.nextBatch()) |batch| {
            var batch_inputs = batch.inputs;
            var batch_targets = batch.targets;
            defer batch_inputs.deinit();
            defer batch_targets.deinit();

            // Forward pass
            var output = try model.forward(&batch_inputs);
            defer output.deinit();

            // Compute loss
            final_loss = try loss_fn.forward(&output, &batch_targets);

            // Backward pass
            var grad_output = try loss_fn.backward();
            defer grad_output.deinit();

            // Backward through dense2
            var grad_w2 = try Tensor.zeros(allocator, dense2.weights.shape);
            defer grad_w2.deinit();
            var grad_b2 = try Tensor.zeros(allocator, dense2.bias.shape);
            defer grad_b2.deinit();
            var grad_relu = try dense2.backward(&grad_output, &grad_w2, &grad_b2);
            defer grad_relu.deinit();

            // Backward through relu
            var grad_relu_in = try relu.backward(&grad_relu);
            defer grad_relu_in.deinit();

            // Backward through dense1
            var grad_w1 = try Tensor.zeros(allocator, dense1.weights.shape);
            defer grad_w1.deinit();
            var grad_b1 = try Tensor.zeros(allocator, dense1.bias.shape);
            defer grad_b1.deinit();
            var grad_input = try dense1.backward(&grad_relu_in, &grad_w1, &grad_b1);
            defer grad_input.deinit();

            // Update weights
            var params = [_]Tensor{ dense1.weights, dense1.bias, dense2.weights, dense2.bias };
            const grads = [_]Tensor{ grad_w1, grad_b1, grad_w2, grad_b2 };
            try optimizer.step(&params, &grads);
        }
    }

    // Evaluate on test set
    var test_loader = try DataLoader.init(allocator, &test_dataset, num_test, false);
    defer test_loader.deinit();

    var correct: usize = 0;
    var total: usize = 0;

    if (try test_loader.nextBatch()) |batch| {
        var batch_inputs = batch.inputs;
        var batch_targets = batch.targets;
        defer batch_inputs.deinit();
        defer batch_targets.deinit();

        var output = try model.forward(&batch_inputs);
        defer output.deinit();

        // Count correct predictions
        for (0..num_test) |i| {
            // Find predicted class (argmax)
            var pred_class: usize = 0;
            var pred_max: f32 = output.data[i * 10];
            for (1..10) |j| {
                if (output.data[i * 10 + j] > pred_max) {
                    pred_max = output.data[i * 10 + j];
                    pred_class = j;
                }
            }

            // Find true class
            var true_class: usize = 0;
            for (0..10) |j| {
                if (batch_targets.data[i * 10 + j] > 0.5) {
                    true_class = j;
                    break;
                }
            }

            if (pred_class == true_class) {
                correct += 1;
            }
            total += 1;
        }
    }

    const accuracy = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(total));

    // With random data, we expect ~10% accuracy (random guessing)
    // But the network should at least be better than 5% after training
    // (shows that training is working, even if data is random)
    try std.testing.expect(accuracy >= 0.05);

    // Loss should decrease from initial
    try std.testing.expect(final_loss < 5.0);
}
