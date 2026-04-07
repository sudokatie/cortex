const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const reduce = @import("../tensor/reduce.zig");

/// Batch Normalization layer.
/// Normalizes inputs to zero mean and unit variance, then applies learnable scale (gamma) and shift (beta).
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
pub const BatchNorm = struct {
    /// Number of features (channels)
    num_features: usize,
    /// Learnable scale parameter (gamma) [num_features]
    gamma: Tensor,
    /// Learnable shift parameter (beta) [num_features]
    beta: Tensor,
    /// Running mean for inference [num_features]
    running_mean: Tensor,
    /// Running variance for inference [num_features]
    running_var: Tensor,
    /// Small constant for numerical stability
    eps: f32,
    /// Momentum for running statistics update
    momentum: f32,
    /// Whether the layer is in training mode
    training: bool,
    /// Cached values for backward pass
    cached_input: ?Tensor,
    cached_mean: ?Tensor,
    cached_var: ?Tensor,
    cached_normalized: ?Tensor,
    /// Allocator
    allocator: Allocator,

    const Self = @This();

    /// Create a new batch normalization layer.
    pub fn init(allocator: Allocator, num_features: usize) !Self {
        var gamma = try Tensor.ones(allocator, &[_]usize{num_features});
        errdefer gamma.deinit();

        var beta = try Tensor.zeros(allocator, &[_]usize{num_features});
        errdefer beta.deinit();

        var running_mean = try Tensor.zeros(allocator, &[_]usize{num_features});
        errdefer running_mean.deinit();

        const running_var = try Tensor.ones(allocator, &[_]usize{num_features});

        return Self{
            .num_features = num_features,
            .gamma = gamma,
            .beta = beta,
            .running_mean = running_mean,
            .running_var = running_var,
            .eps = 1e-5,
            .momentum = 0.1,
            .training = true,
            .cached_input = null,
            .cached_mean = null,
            .cached_var = null,
            .cached_normalized = null,
            .allocator = allocator,
        };
    }

    /// Free all memory.
    pub fn deinit(self: *Self) void {
        self.gamma.deinit();
        self.beta.deinit();
        self.running_mean.deinit();
        self.running_var.deinit();
        if (self.cached_input) |*t| t.deinit();
        if (self.cached_mean) |*t| t.deinit();
        if (self.cached_var) |*t| t.deinit();
        if (self.cached_normalized) |*t| t.deinit();
        self.* = undefined;
    }

    /// Set training mode.
    pub fn train(self: *Self, mode: bool) void {
        self.training = mode;
    }

    /// Forward pass.
    /// Input x: [batch, num_features] or [batch, num_features, height, width]
    /// For simplicity, this implementation handles [batch, features] only.
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        const batch_size = x.shape[0];
        const features = x.shape[1];

        if (features != self.num_features) {
            return error.FeatureMismatch;
        }

        // Clear cached values
        if (self.cached_input) |*t| t.deinit();
        if (self.cached_mean) |*t| t.deinit();
        if (self.cached_var) |*t| t.deinit();
        if (self.cached_normalized) |*t| t.deinit();

        var output = try Tensor.init(self.allocator, x.shape);
        errdefer output.deinit();

        if (self.training) {
            // Compute batch mean and variance
            var batch_mean = try Tensor.zeros(self.allocator, &[_]usize{features});
            errdefer batch_mean.deinit();

            var batch_var = try Tensor.zeros(self.allocator, &[_]usize{features});
            errdefer batch_var.deinit();

            // Mean: sum over batch dimension
            for (0..features) |f| {
                var sum: f32 = 0.0;
                for (0..batch_size) |b| {
                    sum += x.data[b * features + f];
                }
                batch_mean.data[f] = sum / @as(f32, @floatFromInt(batch_size));
            }

            // Variance: sum of squared differences
            for (0..features) |f| {
                var sum_sq: f32 = 0.0;
                const mean = batch_mean.data[f];
                for (0..batch_size) |b| {
                    const diff = x.data[b * features + f] - mean;
                    sum_sq += diff * diff;
                }
                batch_var.data[f] = sum_sq / @as(f32, @floatFromInt(batch_size));
            }

            // Normalize and apply gamma/beta
            var normalized = try Tensor.init(self.allocator, x.shape);
            errdefer normalized.deinit();

            for (0..batch_size) |b| {
                for (0..features) |f| {
                    const idx = b * features + f;
                    const x_val = x.data[idx];
                    const mean = batch_mean.data[f];
                    const variance = batch_var.data[f];
                    const std_dev = @sqrt(variance + self.eps);
                    const norm = (x_val - mean) / std_dev;
                    normalized.data[idx] = norm;
                    output.data[idx] = self.gamma.data[f] * norm + self.beta.data[f];
                }
            }

            // Update running statistics
            for (0..features) |f| {
                self.running_mean.data[f] = (1.0 - self.momentum) * self.running_mean.data[f] +
                    self.momentum * batch_mean.data[f];
                self.running_var.data[f] = (1.0 - self.momentum) * self.running_var.data[f] +
                    self.momentum * batch_var.data[f];
            }

            // Cache for backward
            self.cached_input = try x.clone();
            self.cached_mean = batch_mean;
            self.cached_var = batch_var;
            self.cached_normalized = normalized;
        } else {
            // Inference: use running statistics
            for (0..batch_size) |b| {
                for (0..features) |f| {
                    const idx = b * features + f;
                    const x_val = x.data[idx];
                    const mean = self.running_mean.data[f];
                    const variance = self.running_var.data[f];
                    const std_dev = @sqrt(variance + self.eps);
                    const norm = (x_val - mean) / std_dev;
                    output.data[idx] = self.gamma.data[f] * norm + self.beta.data[f];
                }
            }
        }

        return output;
    }

    /// Backward pass.
    /// grad_output: gradient of loss w.r.t. output [batch, features]
    /// Returns: gradient of loss w.r.t. input [batch, features]
    pub fn backward(self: *Self, grad_output: *const Tensor, grad_gamma: *Tensor, grad_beta: *Tensor) !Tensor {
        const input = &(self.cached_input orelse return error.NoCachedInput);
        const batch_mean = &(self.cached_mean orelse return error.NoCachedMean);
        const batch_var = &(self.cached_var orelse return error.NoCachedVar);
        const normalized = &(self.cached_normalized orelse return error.NoCachedNormalized);

        const batch_size = input.shape[0];
        const features = input.shape[1];
        const n: f32 = @floatFromInt(batch_size);

        var grad_input = try Tensor.zeros(self.allocator, input.shape);
        errdefer grad_input.deinit();

        // Compute gradients for gamma and beta
        // grad_gamma = sum(grad_output * normalized, axis=0)
        // grad_beta = sum(grad_output, axis=0)
        for (0..features) |f| {
            var sum_gamma: f32 = 0.0;
            var sum_beta: f32 = 0.0;
            for (0..batch_size) |b| {
                const idx = b * features + f;
                sum_gamma += grad_output.data[idx] * normalized.data[idx];
                sum_beta += grad_output.data[idx];
            }
            grad_gamma.data[f] = sum_gamma;
            grad_beta.data[f] = sum_beta;
        }

        // Compute gradient w.r.t. input
        // This follows the batch norm backward formula
        for (0..features) |f| {
            const variance = batch_var.data[f];
            const std_dev = @sqrt(variance + self.eps);
            const inv_std = 1.0 / std_dev;
            const gamma = self.gamma.data[f];

            // Compute intermediate sums
            var sum_grad_out: f32 = 0.0;
            var sum_grad_out_x: f32 = 0.0;
            const mean = batch_mean.data[f];

            for (0..batch_size) |b| {
                const idx = b * features + f;
                sum_grad_out += grad_output.data[idx];
                sum_grad_out_x += grad_output.data[idx] * (input.data[idx] - mean);
            }

            // Compute gradient for each element
            for (0..batch_size) |b| {
                const idx = b * features + f;
                const x_centered = input.data[idx] - mean;
                const grad_norm = grad_output.data[idx] * gamma;
                grad_input.data[idx] = inv_std / n * (n * grad_norm - sum_grad_out - x_centered * inv_std * inv_std * sum_grad_out_x);
            }
        }

        return grad_input;
    }
};

// Tests

test "batchnorm forward shape" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm.init(allocator, 4);
    defer bn.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 2, 4 });
    defer x.deinit();

    var y = try bn.forward(&x);
    defer y.deinit();

    try std.testing.expectEqual(@as(usize, 2), y.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), y.shape[1]);
}

test "batchnorm normalizes to zero mean" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm.init(allocator, 2);
    defer bn.deinit();

    // Input with different values per feature
    var x = try Tensor.init(allocator, &[_]usize{ 4, 2 });
    defer x.deinit();
    // Feature 0: [1, 2, 3, 4] mean=2.5, Feature 1: [5, 6, 7, 8] mean=6.5
    x.data[0] = 1.0;
    x.data[1] = 5.0;
    x.data[2] = 2.0;
    x.data[3] = 6.0;
    x.data[4] = 3.0;
    x.data[5] = 7.0;
    x.data[6] = 4.0;
    x.data[7] = 8.0;

    var y = try bn.forward(&x);
    defer y.deinit();

    // Check that output has approximately zero mean per feature
    var mean0: f32 = 0.0;
    var mean1: f32 = 0.0;
    for (0..4) |b| {
        mean0 += y.data[b * 2 + 0];
        mean1 += y.data[b * 2 + 1];
    }
    mean0 /= 4.0;
    mean1 /= 4.0;

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean0, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean1, 0.01);
}

test "batchnorm inference uses running stats" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm.init(allocator, 2);
    defer bn.deinit();

    // Set known running statistics
    bn.running_mean.data[0] = 2.0;
    bn.running_mean.data[1] = 4.0;
    bn.running_var.data[0] = 1.0;
    bn.running_var.data[1] = 4.0;

    // Switch to eval mode
    bn.train(false);

    var x = try Tensor.init(allocator, &[_]usize{ 1, 2 });
    defer x.deinit();
    x.data[0] = 3.0; // (3 - 2) / sqrt(1 + eps) ≈ 1.0
    x.data[1] = 6.0; // (6 - 4) / sqrt(4 + eps) ≈ 1.0

    var y = try bn.forward(&x);
    defer y.deinit();

    // With gamma=1 and beta=0, output should be normalized values
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y.data[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y.data[1], 0.01);
}

test "batchnorm backward gradients" {
    const allocator = std.testing.allocator;

    var bn = try BatchNorm.init(allocator, 2);
    defer bn.deinit();

    var x = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer x.deinit();
    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = 3.0;
    x.data[3] = 4.0;

    var y = try bn.forward(&x);
    defer y.deinit();

    var grad_out = try Tensor.ones(allocator, &[_]usize{ 2, 2 });
    defer grad_out.deinit();

    var grad_gamma = try Tensor.zeros(allocator, &[_]usize{2});
    defer grad_gamma.deinit();

    var grad_beta = try Tensor.zeros(allocator, &[_]usize{2});
    defer grad_beta.deinit();

    var grad_input = try bn.backward(&grad_out, &grad_gamma, &grad_beta);
    defer grad_input.deinit();

    // grad_beta should be sum of grad_output per feature
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_beta.data[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_beta.data[1], 0.01);

    // grad_input should exist and have correct shape
    try std.testing.expectEqual(@as(usize, 2), grad_input.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), grad_input.shape[1]);
}
