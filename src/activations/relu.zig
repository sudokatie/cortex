const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// ReLU activation: max(0, x)
pub const ReLU = struct {
    allocator: Allocator,
    cached_input: ?Tensor,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .cached_input = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.cached_input) |*ci| {
            ci.deinit();
        }
        self.* = undefined;
    }

    /// Forward pass: y = max(0, x)
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        // Cache input for backward
        if (self.cached_input) |*ci| {
            ci.deinit();
        }
        self.cached_input = try x.clone();

        const result = try Tensor.init(self.allocator, x.shape);
        for (result.data, x.data) |*out, in| {
            out.* = @max(0.0, in);
        }
        return result;
    }

    /// Backward pass: grad_input = grad_output * (x > 0 ? 1 : 0)
    pub fn backward(self: *Self, grad_output: *const Tensor) !Tensor {
        const input = &(self.cached_input orelse return error.NoCachedInput);

        const grad_input = try Tensor.init(self.allocator, grad_output.shape);
        for (grad_input.data, grad_output.data, input.data) |*gi, go, x| {
            gi.* = if (x > 0.0) go else 0.0;
        }
        return grad_input;
    }
};

/// Functional ReLU (no caching)
pub fn relu(allocator: Allocator, x: *const Tensor) !Tensor {
    const result = try Tensor.init(allocator, x.shape);
    for (result.data, x.data) |*out, in| {
        out.* = @max(0.0, in);
    }
    return result;
}

/// Functional ReLU gradient
pub fn reluGrad(allocator: Allocator, grad_output: *const Tensor, x: *const Tensor) !Tensor {
    const grad = try Tensor.init(allocator, grad_output.shape);
    for (grad.data, grad_output.data, x.data) |*g, go, xv| {
        g.* = if (xv > 0.0) go else 0.0;
    }
    return grad;
}

// Tests

test "relu forward" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{5});
    defer x.deinit();
    x.data[0] = -2.0;
    x.data[1] = -1.0;
    x.data[2] = 0.0;
    x.data[3] = 1.0;
    x.data[4] = 2.0;

    var activation = ReLU.init(allocator);
    defer activation.deinit();

    var y = try activation.forward(&x);
    defer y.deinit();

    try std.testing.expectEqual(@as(f32, 0.0), y.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), y.data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), y.data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), y.data[3]);
    try std.testing.expectEqual(@as(f32, 2.0), y.data[4]);
}

test "relu backward" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{4});
    defer x.deinit();
    x.data[0] = -1.0;
    x.data[1] = 0.0;
    x.data[2] = 1.0;
    x.data[3] = 2.0;

    var activation = ReLU.init(allocator);
    defer activation.deinit();

    var y = try activation.forward(&x);
    defer y.deinit();

    var grad_out = try Tensor.ones(allocator, &[_]usize{4});
    defer grad_out.deinit();

    var grad_in = try activation.backward(&grad_out);
    defer grad_in.deinit();

    // grad = 0 where x <= 0, 1 where x > 0
    try std.testing.expectEqual(@as(f32, 0.0), grad_in.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), grad_in.data[1]);
    try std.testing.expectEqual(@as(f32, 1.0), grad_in.data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), grad_in.data[3]);
}

test "relu functional" {
    const allocator = std.testing.allocator;

    var x = try Tensor.init(allocator, &[_]usize{3});
    defer x.deinit();
    x.data[0] = -5.0;
    x.data[1] = 0.0;
    x.data[2] = 5.0;

    var y = try relu(allocator, &x);
    defer y.deinit();

    try std.testing.expectEqual(@as(f32, 0.0), y.data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), y.data[1]);
    try std.testing.expectEqual(@as(f32, 5.0), y.data[2]);
}
