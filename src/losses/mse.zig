const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Mean Squared Error loss: mean((pred - target)^2)
pub const MSELoss = struct {
    allocator: Allocator,
    cached_pred: ?Tensor,
    cached_target: ?Tensor,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .cached_pred = null,
            .cached_target = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.cached_pred) |*cp| {
            cp.deinit();
        }
        if (self.cached_target) |*ct| {
            ct.deinit();
        }
        self.* = undefined;
    }

    /// Forward pass: compute MSE loss
    /// pred, target: same shape tensors
    /// Returns: scalar loss value
    pub fn forward(self: *Self, pred: *const Tensor, target: *const Tensor) !f32 {
        if (!pred.sameShape(target)) {
            return error.ShapeMismatch;
        }

        // Cache for backward
        if (self.cached_pred) |*cp| {
            cp.deinit();
        }
        self.cached_pred = try pred.clone();

        if (self.cached_target) |*ct| {
            ct.deinit();
        }
        self.cached_target = try target.clone();

        // Compute MSE
        var sum_sq: f32 = 0.0;
        for (pred.data, target.data) |p, t| {
            const diff = p - t;
            sum_sq += diff * diff;
        }

        return sum_sq / @as(f32, @floatFromInt(pred.data.len));
    }

    /// Backward pass: gradient w.r.t. predictions
    /// grad = 2 * (pred - target) / n
    pub fn backward(self: *Self) !Tensor {
        const pred = &(self.cached_pred orelse return error.NoCachedPred);
        const target = &(self.cached_target orelse return error.NoCachedTarget);

        const n = @as(f32, @floatFromInt(pred.data.len));
        const grad = try Tensor.init(self.allocator, pred.shape);

        for (grad.data, pred.data, target.data) |*g, p, t| {
            g.* = 2.0 * (p - t) / n;
        }

        return grad;
    }
};

/// Functional MSE loss (no caching)
pub fn mseLoss(pred: *const Tensor, target: *const Tensor) !f32 {
    if (!pred.sameShape(target)) {
        return error.ShapeMismatch;
    }

    var sum_sq: f32 = 0.0;
    for (pred.data, target.data) |p, t| {
        const diff = p - t;
        sum_sq += diff * diff;
    }

    return sum_sq / @as(f32, @floatFromInt(pred.data.len));
}

/// Functional MSE gradient
pub fn mseGrad(allocator: Allocator, pred: *const Tensor, target: *const Tensor) !Tensor {
    if (!pred.sameShape(target)) {
        return error.ShapeMismatch;
    }

    const n = @as(f32, @floatFromInt(pred.data.len));
    const grad = try Tensor.init(allocator, pred.shape);

    for (grad.data, pred.data, target.data) |*g, p, t| {
        g.* = 2.0 * (p - t) / n;
    }

    return grad;
}

// Tests

test "mse zero loss" {
    const allocator = std.testing.allocator;

    var pred = try Tensor.init(allocator, &[_]usize{4});
    defer pred.deinit();
    pred.data[0] = 1.0;
    pred.data[1] = 2.0;
    pred.data[2] = 3.0;
    pred.data[3] = 4.0;

    var target = try Tensor.init(allocator, &[_]usize{4});
    defer target.deinit();
    target.data[0] = 1.0;
    target.data[1] = 2.0;
    target.data[2] = 3.0;
    target.data[3] = 4.0;

    var loss_fn = MSELoss.init(allocator);
    defer loss_fn.deinit();

    const loss = try loss_fn.forward(&pred, &target);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), loss, 0.001);
}

test "mse known value" {
    const allocator = std.testing.allocator;

    // pred = [1, 2], target = [2, 4]
    // diff = [-1, -2], diff^2 = [1, 4]
    // mse = (1 + 4) / 2 = 2.5
    var pred = try Tensor.init(allocator, &[_]usize{2});
    defer pred.deinit();
    pred.data[0] = 1.0;
    pred.data[1] = 2.0;

    var target = try Tensor.init(allocator, &[_]usize{2});
    defer target.deinit();
    target.data[0] = 2.0;
    target.data[1] = 4.0;

    const loss = try mseLoss(&pred, &target);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), loss, 0.001);
}

test "mse gradient" {
    const allocator = std.testing.allocator;

    var pred = try Tensor.init(allocator, &[_]usize{2});
    defer pred.deinit();
    pred.data[0] = 3.0;
    pred.data[1] = 1.0;

    var target = try Tensor.init(allocator, &[_]usize{2});
    defer target.deinit();
    target.data[0] = 1.0;
    target.data[1] = 1.0;

    var loss_fn = MSELoss.init(allocator);
    defer loss_fn.deinit();

    _ = try loss_fn.forward(&pred, &target);

    var grad = try loss_fn.backward();
    defer grad.deinit();

    // grad = 2 * (pred - target) / n
    // grad[0] = 2 * (3 - 1) / 2 = 2
    // grad[1] = 2 * (1 - 1) / 2 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), grad.data[1], 0.001);
}

test "mse functional gradient" {
    const allocator = std.testing.allocator;

    var pred = try Tensor.init(allocator, &[_]usize{4});
    defer pred.deinit();
    for (0..4) |i| {
        pred.data[i] = @floatFromInt(i);
    }

    var target = try Tensor.zeros(allocator, &[_]usize{4});
    defer target.deinit();

    var grad = try mseGrad(allocator, &pred, &target);
    defer grad.deinit();

    // pred = [0, 1, 2, 3], target = [0, 0, 0, 0]
    // grad = 2 * [0, 1, 2, 3] / 4 = [0, 0.5, 1.0, 1.5]
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), grad.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), grad.data[3], 0.001);
}
