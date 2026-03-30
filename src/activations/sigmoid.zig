const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Sigmoid activation: 1 / (1 + exp(-x))
pub const Sigmoid = struct {
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

    /// Forward pass: y = 1 / (1 + exp(-x))
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        const result = try Tensor.init(self.allocator, x.shape);
        for (result.data, x.data) |*out, in| {
            out.* = 1.0 / (1.0 + @exp(-in));
        }

        // Cache output for backward (sigmoid grad uses output, not input)
        if (self.cached_output) |*co| {
            co.deinit();
        }
        self.cached_output = try result.clone();

        return result;
    }

    /// Backward pass: grad_input = grad_output * sigmoid * (1 - sigmoid)
    pub fn backward(self: *Self, grad_output: *const Tensor) !Tensor {
        const output = &(self.cached_output orelse return error.NoCachedOutput);

        const grad_input = try Tensor.init(self.allocator, grad_output.shape);
        for (grad_input.data, grad_output.data, output.data) |*gi, go, sig| {
            gi.* = go * sig * (1.0 - sig);
        }
        return grad_input;
    }
};

/// Functional sigmoid (no caching)
pub fn sigmoid(allocator: Allocator, x: *const Tensor) !Tensor {
    const result = try Tensor.init(allocator, x.shape);
    for (result.data, x.data) |*out, in| {
        out.* = 1.0 / (1.0 + @exp(-in));
    }
    return result;
}

/// Functional sigmoid gradient
pub fn sigmoidGrad(allocator: Allocator, grad_output: *const Tensor, output: *const Tensor) !Tensor {
    const grad = try Tensor.init(allocator, grad_output.shape);
    for (grad.data, grad_output.data, output.data) |*g, go, sig| {
        g.* = go * sig * (1.0 - sig);
    }
    return grad;
}

// Tests

test "sigmoid forward" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{3});
    defer x.deinit();
    x.data[0] = 0.0;
    x.data[1] = -10.0;
    x.data[2] = 10.0;

    var activation = Sigmoid.init(allocator);
    defer activation.deinit();

    var y = try activation.forward(&x);
    defer y.deinit();

    // sigmoid(0) = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), y.data[0], 0.001);
    // sigmoid(-10) ≈ 0
    try std.testing.expect(y.data[1] < 0.001);
    // sigmoid(10) ≈ 1
    try std.testing.expect(y.data[2] > 0.999);
}

test "sigmoid backward" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{1});
    defer x.deinit();
    x.data[0] = 0.0; // sigmoid(0) = 0.5

    var activation = Sigmoid.init(allocator);
    defer activation.deinit();

    var y = try activation.forward(&x);
    defer y.deinit();

    var grad_out = try Tensor.ones(allocator, &[_]usize{1});
    defer grad_out.deinit();

    var grad_in = try activation.backward(&grad_out);
    defer grad_in.deinit();

    // grad = sigmoid * (1 - sigmoid) = 0.5 * 0.5 = 0.25
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), grad_in.data[0], 0.001);
}

test "sigmoid range" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{100});
    defer x.deinit();
    for (0..100) |i| {
        x.data[i] = @as(f32, @floatFromInt(i)) - 50.0; // -50 to 49
    }

    var y = try sigmoid(allocator, &x);
    defer y.deinit();

    // All outputs should be in [0, 1]
    for (y.data) |v| {
        try std.testing.expect(v >= 0.0);
        try std.testing.expect(v <= 1.0);
    }
}
