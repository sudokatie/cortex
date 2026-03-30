const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Tanh activation: (e^x - e^-x) / (e^x + e^-x)
pub const Tanh = struct {
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

    /// Forward pass: y = tanh(x)
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        const result = try Tensor.init(self.allocator, x.shape);
        for (result.data, x.data) |*out, in| {
            out.* = std.math.tanh(in);
        }

        // Cache output for backward (tanh grad uses output)
        if (self.cached_output) |*co| {
            co.deinit();
        }
        self.cached_output = try result.clone();

        return result;
    }

    /// Backward pass: grad_input = grad_output * (1 - tanh^2)
    pub fn backward(self: *Self, grad_output: *const Tensor) !Tensor {
        const output = &(self.cached_output orelse return error.NoCachedOutput);

        const grad_input = try Tensor.init(self.allocator, grad_output.shape);
        for (grad_input.data, grad_output.data, output.data) |*gi, go, t| {
            gi.* = go * (1.0 - t * t);
        }
        return grad_input;
    }
};

/// Functional tanh (no caching)
pub fn tanh_activation(allocator: Allocator, x: *const Tensor) !Tensor {
    const result = try Tensor.init(allocator, x.shape);
    for (result.data, x.data) |*out, in| {
        out.* = std.math.tanh(in);
    }
    return result;
}

/// Functional tanh gradient
pub fn tanhGrad(allocator: Allocator, grad_output: *const Tensor, output: *const Tensor) !Tensor {
    const grad = try Tensor.init(allocator, grad_output.shape);
    for (grad.data, grad_output.data, output.data) |*g, go, t| {
        g.* = go * (1.0 - t * t);
    }
    return grad;
}

// Tests

test "tanh forward" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{3});
    defer x.deinit();
    x.data[0] = 0.0;
    x.data[1] = -10.0;
    x.data[2] = 10.0;

    var activation = Tanh.init(allocator);
    defer activation.deinit();

    var y = try activation.forward(&x);
    defer y.deinit();

    // tanh(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y.data[0], 0.001);
    // tanh(-10) ≈ -1
    try std.testing.expect(y.data[1] < -0.999);
    // tanh(10) ≈ 1
    try std.testing.expect(y.data[2] > 0.999);
}

test "tanh backward" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{1});
    defer x.deinit();
    x.data[0] = 0.0; // tanh(0) = 0

    var activation = Tanh.init(allocator);
    defer activation.deinit();

    var y = try activation.forward(&x);
    defer y.deinit();

    var grad_out = try Tensor.ones(allocator, &[_]usize{1});
    defer grad_out.deinit();

    var grad_in = try activation.backward(&grad_out);
    defer grad_in.deinit();

    // grad = 1 - tanh^2(0) = 1 - 0 = 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_in.data[0], 0.001);
}

test "tanh range" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{100});
    defer x.deinit();
    for (0..100) |i| {
        x.data[i] = @as(f32, @floatFromInt(i)) - 50.0; // -50 to 49
    }

    var y = try tanh_activation(allocator, &x);
    defer y.deinit();

    // All outputs should be in [-1, 1]
    for (y.data) |v| {
        try std.testing.expect(v >= -1.0);
        try std.testing.expect(v <= 1.0);
    }
}
