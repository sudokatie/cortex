const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Softmax activation: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
/// Subtracting max for numerical stability.
pub const Softmax = struct {
    allocator: Allocator,
    cached_output: ?Tensor,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .cached_output = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.cached_output) |*co| {
            co.deinit();
        }
        self.* = undefined;
    }

    /// Forward pass for 2D input [batch, classes]
    /// Softmax applied along last axis (axis=1)
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        if (x.shape.len != 2) {
            return error.InvalidShape;
        }

        const batch_size = x.shape[0];
        const num_classes = x.shape[1];

        var result = try Tensor.init(self.allocator, x.shape);
        errdefer result.deinit();

        for (0..batch_size) |b| {
            // Find max for numerical stability
            var max_val: f32 = x.data[b * num_classes];
            for (1..num_classes) |c| {
                const val = x.data[b * num_classes + c];
                if (val > max_val) max_val = val;
            }

            // Compute exp(x - max) and sum
            var sum: f32 = 0.0;
            for (0..num_classes) |c| {
                const idx = b * num_classes + c;
                const exp_val = @exp(x.data[idx] - max_val);
                result.data[idx] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for (0..num_classes) |c| {
                const idx = b * num_classes + c;
                result.data[idx] /= sum;
            }
        }

        // Cache for backward
        if (self.cached_output) |*co| {
            co.deinit();
        }
        self.cached_output = try result.clone();

        return result;
    }

    /// Backward pass (full Jacobian, not combined with cross-entropy)
    /// For cross-entropy, use combined gradient: softmax - y_true
    pub fn backward(self: *Self, grad_output: *const Tensor) !Tensor {
        const output = &(self.cached_output orelse return error.NoCachedOutput);

        if (grad_output.shape.len != 2) {
            return error.InvalidShape;
        }

        const batch_size = grad_output.shape[0];
        const num_classes = grad_output.shape[1];

        var grad_input = try Tensor.zeros(self.allocator, grad_output.shape);
        errdefer grad_input.deinit();

        // Softmax Jacobian: dL/dx_i = sum_j(dL/dy_j * dy_j/dx_i)
        // dy_j/dx_i = y_i * (delta_ij - y_j)
        for (0..batch_size) |b| {
            for (0..num_classes) |i| {
                var sum: f32 = 0.0;
                for (0..num_classes) |j| {
                    const y_i = output.data[b * num_classes + i];
                    const y_j = output.data[b * num_classes + j];
                    const grad_j = grad_output.data[b * num_classes + j];

                    const delta_ij: f32 = if (i == j) 1.0 else 0.0;
                    sum += grad_j * y_i * (delta_ij - y_j);
                }
                grad_input.data[b * num_classes + i] = sum;
            }
        }

        return grad_input;
    }
};

/// Functional softmax (no caching)
pub fn softmax(allocator: Allocator, x: *const Tensor) !Tensor {
    if (x.shape.len != 2) {
        return error.InvalidShape;
    }

    const batch_size = x.shape[0];
    const num_classes = x.shape[1];

    var result = try Tensor.init(allocator, x.shape);

    for (0..batch_size) |b| {
        var max_val: f32 = x.data[b * num_classes];
        for (1..num_classes) |c| {
            const val = x.data[b * num_classes + c];
            if (val > max_val) max_val = val;
        }

        var sum: f32 = 0.0;
        for (0..num_classes) |c| {
            const idx = b * num_classes + c;
            const exp_val = @exp(x.data[idx] - max_val);
            result.data[idx] = exp_val;
            sum += exp_val;
        }

        for (0..num_classes) |c| {
            const idx = b * num_classes + c;
            result.data[idx] /= sum;
        }
    }

    return result;
}

// Tests

test "softmax sums to 1" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{ 2, 4 });
    defer x.deinit();
    // Row 0: [1, 2, 3, 4]
    // Row 1: [0, 0, 0, 0]
    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = 3.0;
    x.data[3] = 4.0;
    x.data[4] = 0.0;
    x.data[5] = 0.0;
    x.data[6] = 0.0;
    x.data[7] = 0.0;

    var activation = Softmax.init(allocator);
    defer activation.deinit();

    var y = try activation.forward(&x);
    defer y.deinit();

    // Sum of each row should be 1
    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    for (0..4) |c| {
        sum0 += y.data[c];
        sum1 += y.data[4 + c];
    }

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum0, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum1, 0.001);
}

test "softmax values in 0-1" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{ 1, 3 });
    defer x.deinit();
    x.data[0] = -100.0;
    x.data[1] = 0.0;
    x.data[2] = 100.0;

    var y = try softmax(allocator, &x);
    defer y.deinit();

    for (y.data) |v| {
        try std.testing.expect(v >= 0.0);
        try std.testing.expect(v <= 1.0);
    }
}

test "softmax uniform input" {
    const allocator = std.testing.allocator;

    // When all inputs are equal, softmax should be uniform
    var x = try Tensor.init(allocator, &[_]usize{ 1, 4 });
    defer x.deinit();
    for (x.data) |*v| {
        v.* = 5.0;
    }

    var y = try softmax(allocator, &x);
    defer y.deinit();

    // Each output should be 0.25
    for (y.data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), v, 0.001);
    }
}

test "softmax backward" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{ 1, 3 });
    defer x.deinit();
    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = 3.0;

    var activation = Softmax.init(allocator);
    defer activation.deinit();

    var y = try activation.forward(&x);
    defer y.deinit();

    var grad_out = try Tensor.ones(allocator, &[_]usize{ 1, 3 });
    defer grad_out.deinit();

    var grad_in = try activation.backward(&grad_out);
    defer grad_in.deinit();

    // With uniform grad_output of 1s, the sum of grad_input should be 0
    // because softmax gradients sum to 0 when output grad is uniform
    var sum: f32 = 0.0;
    for (grad_in.data) |g| {
        sum += g;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum, 0.01);
}
