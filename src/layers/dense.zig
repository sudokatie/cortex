const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const matmul = @import("../tensor/matmul.zig");
const reduce = @import("../tensor/reduce.zig");
const broadcast = @import("../tensor/broadcast.zig");
const ops = @import("../tensor/ops.zig");

/// Fully connected (dense) layer: y = x @ W + b
/// Where x is [batch, in_features], W is [in_features, out_features], b is [out_features]
pub const Dense = struct {
    /// Weight matrix [in_features, out_features]
    weights: Tensor,
    /// Bias vector [out_features]
    bias: Tensor,
    /// Allocator for creating tensors
    allocator: Allocator,
    /// Cached input for backward pass
    cached_input: ?Tensor,

    const Self = @This();

    /// Create a new dense layer.
    /// Weights are initialized to small random values, bias to zeros.
    pub fn init(allocator: Allocator, in_features: usize, out_features: usize) !Self {
        var weights = try Tensor.init(allocator, &[_]usize{ in_features, out_features });
        errdefer weights.deinit();

        // Initialize weights to small values (Xavier-like initialization)
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(in_features)));
        var prng = std.Random.DefaultPrng.init(0);
        const rand = prng.random();
        for (weights.data) |*w| {
            w.* = (rand.float(f32) * 2.0 - 1.0) * scale;
        }

        const bias = try Tensor.zeros(allocator, &[_]usize{out_features});

        return Self{
            .weights = weights,
            .bias = bias,
            .allocator = allocator,
            .cached_input = null,
        };
    }

    /// Free all memory.
    pub fn deinit(self: *Self) void {
        self.weights.deinit();
        self.bias.deinit();
        if (self.cached_input) |*ci| {
            ci.deinit();
        }
        self.* = undefined;
    }

    /// Forward pass: y = x @ W + b
    /// Input x: [batch, in_features]
    /// Output: [batch, out_features]
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        // Cache input for backward pass
        if (self.cached_input) |*ci| {
            ci.deinit();
        }
        self.cached_input = try x.clone();

        // y = x @ W (matmul)
        var y = try matmul.matmul(self.allocator, x, &self.weights);
        errdefer y.deinit();

        // Add bias (broadcast [out_features] to [batch, out_features])
        const batch_size = x.shape[0];
        const out_features = self.bias.shape[0];

        for (0..batch_size) |b| {
            for (0..out_features) |o| {
                const idx = b * out_features + o;
                y.data[idx] += self.bias.data[o];
            }
        }

        return y;
    }

    /// Backward pass: compute gradients.
    /// grad_output: gradient of loss w.r.t. output [batch, out_features]
    /// Returns: gradient of loss w.r.t. input [batch, in_features]
    pub fn backward(self: *Self, grad_output: *const Tensor, grad_weights: *Tensor, grad_bias: *Tensor) !Tensor {
        const input = &(self.cached_input orelse return error.NoCachedInput);

        // dW = x.T @ grad_output
        // input: [batch, in_features], grad_output: [batch, out_features]
        // dW: [in_features, out_features]
        var dw = try matmul.matmulTransposeA(self.allocator, input, grad_output);
        defer dw.deinit();

        @memcpy(grad_weights.data, dw.data);

        // db = sum(grad_output, axis=0) -> [out_features]
        var db = try reduce.sum(self.allocator, grad_output, 0);
        defer db.deinit();

        @memcpy(grad_bias.data, db.data);

        // dx = grad_output @ W.T
        // grad_output: [batch, out_features], W: [in_features, out_features]
        // dx: [batch, in_features]
        const dx = try matmul.matmulTransposeB(self.allocator, grad_output, &self.weights);

        return dx;
    }

    /// Get number of input features.
    pub fn inFeatures(self: *const Self) usize {
        return self.weights.shape[0];
    }

    /// Get number of output features.
    pub fn outFeatures(self: *const Self) usize {
        return self.weights.shape[1];
    }
};

// Tests

test "dense forward shape" {
    const allocator = std.testing.allocator;

    var layer = try Dense.init(allocator, 4, 3);
    defer layer.deinit();

    // Input: [2, 4] (batch=2, in_features=4)
    var x = try Tensor.ones(allocator, &[_]usize{ 2, 4 });
    defer x.deinit();

    var y = try layer.forward(&x);
    defer y.deinit();

    // Output should be [2, 3]
    try std.testing.expectEqual(@as(usize, 2), y.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), y.shape[1]);
}

test "dense forward values" {
    const allocator = std.testing.allocator;

    var layer = try Dense.init(allocator, 2, 2);
    defer layer.deinit();

    // Set known weights and bias for verification
    layer.weights.data[0] = 1.0; // W[0,0]
    layer.weights.data[1] = 2.0; // W[0,1]
    layer.weights.data[2] = 3.0; // W[1,0]
    layer.weights.data[3] = 4.0; // W[1,1]
    layer.bias.data[0] = 0.5;
    layer.bias.data[1] = 0.5;

    // Input: [1, 2]
    var x = try Tensor.init(allocator, &[_]usize{ 1, 2 });
    defer x.deinit();
    x.data[0] = 1.0;
    x.data[1] = 1.0;

    var y = try layer.forward(&x);
    defer y.deinit();

    // y = [1,1] @ [[1,2],[3,4]] + [0.5, 0.5]
    // y = [1*1+1*3, 1*2+1*4] + [0.5, 0.5]
    // y = [4, 6] + [0.5, 0.5] = [4.5, 6.5]
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), y.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.5), y.data[1], 0.001);
}

test "dense backward gradients" {
    const allocator = std.testing.allocator;

    var layer = try Dense.init(allocator, 2, 2);
    defer layer.deinit();

    // Set weights
    layer.weights.data[0] = 1.0;
    layer.weights.data[1] = 2.0;
    layer.weights.data[2] = 3.0;
    layer.weights.data[3] = 4.0;

    // Forward pass
    var x = try Tensor.init(allocator, &[_]usize{ 1, 2 });
    defer x.deinit();
    x.data[0] = 1.0;
    x.data[1] = 2.0;

    var y = try layer.forward(&x);
    defer y.deinit();

    // Gradient of loss w.r.t. output (pretend loss = sum(y))
    var grad_out = try Tensor.ones(allocator, &[_]usize{ 1, 2 });
    defer grad_out.deinit();

    // Allocate gradient tensors
    var grad_weights = try Tensor.zeros(allocator, &[_]usize{ 2, 2 });
    defer grad_weights.deinit();

    var grad_bias = try Tensor.zeros(allocator, &[_]usize{2});
    defer grad_bias.deinit();

    var grad_input = try layer.backward(&grad_out, &grad_weights, &grad_bias);
    defer grad_input.deinit();

    // dW = x.T @ grad_out = [[1],[2]] @ [[1,1]] = [[1,1],[2,2]]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_weights.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_weights.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_weights.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_weights.data[3], 0.001);

    // db = sum(grad_out, axis=0) = [1, 1]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_bias.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_bias.data[1], 0.001);

    // dx = grad_out @ W.T = [[1,1]] @ [[1,3],[2,4]] = [1*1+1*2, 1*3+1*4] = [3, 7]
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grad_input.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), grad_input.data[1], 0.001);
}

test "dense features" {
    const allocator = std.testing.allocator;

    var layer = try Dense.init(allocator, 10, 5);
    defer layer.deinit();

    try std.testing.expectEqual(@as(usize, 10), layer.inFeatures());
    try std.testing.expectEqual(@as(usize, 5), layer.outFeatures());
}
