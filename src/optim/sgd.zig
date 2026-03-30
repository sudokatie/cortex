const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// SGD optimizer with momentum
/// Update rule: velocity = momentum * velocity - lr * grad
///              param += velocity
pub const SGD = struct {
    allocator: Allocator,
    lr: f32,
    momentum: f32,
    velocities: std.ArrayListUnmanaged(Tensor),

    const Self = @This();

    pub fn init(allocator: Allocator, lr: f32, momentum: f32) Self {
        return Self{
            .allocator = allocator,
            .lr = lr,
            .momentum = momentum,
            .velocities = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.velocities.items) |*v| {
            v.deinit();
        }
        self.velocities.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register a parameter tensor for optimization.
    /// Must be called before step() for each parameter.
    pub fn addParam(self: *Self, param: *const Tensor) !void {
        const velocity = try Tensor.zeros(self.allocator, param.shape);
        try self.velocities.append(self.allocator, velocity);
    }

    /// Perform one optimization step.
    /// params: slice of parameter tensors
    /// grads: slice of gradient tensors (same order as params)
    pub fn step(self: *Self, params: []Tensor, grads: []const Tensor) !void {
        if (params.len != grads.len) {
            return error.ParamGradMismatch;
        }
        if (params.len != self.velocities.items.len) {
            return error.VelocityMismatch;
        }

        for (params, grads, self.velocities.items) |*param, grad, *velocity| {
            if (param.data.len != grad.data.len) {
                return error.ShapeMismatch;
            }

            for (param.data, grad.data, velocity.data) |*p, g, *v| {
                // velocity = momentum * velocity - lr * grad
                v.* = self.momentum * v.* - self.lr * g;
                // param += velocity
                p.* += v.*;
            }
        }
    }

    /// Zero out all velocities
    pub fn zeroVelocity(self: *Self) void {
        for (self.velocities.items) |*v| {
            v.fill(0.0);
        }
    }
};

// Tests

test "sgd basic step" {
    const allocator = std.testing.allocator;

    var optimizer = SGD.init(allocator, 0.1, 0.0);
    defer optimizer.deinit();

    // Create parameter
    var param = try Tensor.init(allocator, &[_]usize{3});
    defer param.deinit();
    param.data[0] = 1.0;
    param.data[1] = 2.0;
    param.data[2] = 3.0;

    // Register param
    try optimizer.addParam(&param);

    // Create gradient
    var grad = try Tensor.init(allocator, &[_]usize{3});
    defer grad.deinit();
    grad.data[0] = 1.0;
    grad.data[1] = 1.0;
    grad.data[2] = 1.0;

    // Step
    var params = [_]Tensor{param};
    const grads = [_]Tensor{grad};
    try optimizer.step(&params, &grads);
    param = params[0];

    // With lr=0.1, momentum=0, param -= 0.1 * grad
    // [1, 2, 3] - 0.1 * [1, 1, 1] = [0.9, 1.9, 2.9]
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), param.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9), param.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.9), param.data[2], 0.001);
}

test "sgd with momentum" {
    const allocator = std.testing.allocator;

    var optimizer = SGD.init(allocator, 0.1, 0.9);
    defer optimizer.deinit();

    var param = try Tensor.init(allocator, &[_]usize{1});
    defer param.deinit();
    param.data[0] = 10.0;

    try optimizer.addParam(&param);

    var grad = try Tensor.ones(allocator, &[_]usize{1});
    defer grad.deinit();

    // First step: velocity = 0.9 * 0 - 0.1 * 1 = -0.1
    // param = 10 + (-0.1) = 9.9
    var params = [_]Tensor{param};
    const grads = [_]Tensor{grad};
    try optimizer.step(&params, &grads);
    param = params[0];

    try std.testing.expectApproxEqAbs(@as(f32, 9.9), param.data[0], 0.001);

    // Second step: velocity = 0.9 * (-0.1) - 0.1 * 1 = -0.09 - 0.1 = -0.19
    // param = 9.9 + (-0.19) = 9.71
    params = [_]Tensor{param};
    try optimizer.step(&params, &grads);
    param = params[0];

    try std.testing.expectApproxEqAbs(@as(f32, 9.71), param.data[0], 0.001);
}

test "sgd multiple params" {
    const allocator = std.testing.allocator;

    var optimizer = SGD.init(allocator, 0.5, 0.0);
    defer optimizer.deinit();

    var param1 = try Tensor.ones(allocator, &[_]usize{2});
    defer param1.deinit();

    var param2 = try Tensor.init(allocator, &[_]usize{2});
    defer param2.deinit();
    param2.data[0] = 5.0;
    param2.data[1] = 5.0;

    try optimizer.addParam(&param1);
    try optimizer.addParam(&param2);

    var grad1 = try Tensor.ones(allocator, &[_]usize{2});
    defer grad1.deinit();

    var grad2 = try Tensor.init(allocator, &[_]usize{2});
    defer grad2.deinit();
    grad2.data[0] = 2.0;
    grad2.data[1] = 2.0;

    var params = [_]Tensor{ param1, param2 };
    const grads = [_]Tensor{ grad1, grad2 };
    try optimizer.step(&params, &grads);

    // param1: [1, 1] - 0.5 * [1, 1] = [0.5, 0.5]
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), params[0].data[0], 0.001);

    // param2: [5, 5] - 0.5 * [2, 2] = [4, 4]
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), params[1].data[0], 0.001);
}

test "sgd zero velocity" {
    const allocator = std.testing.allocator;

    var optimizer = SGD.init(allocator, 0.1, 0.9);
    defer optimizer.deinit();

    var param = try Tensor.ones(allocator, &[_]usize{2});
    defer param.deinit();

    try optimizer.addParam(&param);

    var grad = try Tensor.ones(allocator, &[_]usize{2});
    defer grad.deinit();

    // Take a step to build up velocity
    var params = [_]Tensor{param};
    const grads = [_]Tensor{grad};
    try optimizer.step(&params, &grads);

    // Zero velocity
    optimizer.zeroVelocity();

    // Velocity should be 0
    for (optimizer.velocities.items[0].data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}
