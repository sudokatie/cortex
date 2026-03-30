const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Adam optimizer
/// Update rule:
///   m = beta1 * m + (1 - beta1) * g
///   v = beta2 * v + (1 - beta2) * g^2
///   m_hat = m / (1 - beta1^t)
///   v_hat = v / (1 - beta2^t)
///   param -= lr * m_hat / (sqrt(v_hat) + eps)
pub const Adam = struct {
    allocator: Allocator,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize, // timestep
    m: std.ArrayListUnmanaged(Tensor), // first moment
    v: std.ArrayListUnmanaged(Tensor), // second moment

    const Self = @This();

    pub fn init(allocator: Allocator, lr: f32) Self {
        return initWithParams(allocator, lr, 0.9, 0.999, 1e-8);
    }

    pub fn initWithParams(allocator: Allocator, lr: f32, beta1: f32, beta2: f32, eps: f32) Self {
        return Self{
            .allocator = allocator,
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .eps = eps,
            .t = 0,
            .m = .{},
            .v = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.m.items) |*tensor| {
            tensor.deinit();
        }
        self.m.deinit(self.allocator);

        for (self.v.items) |*tensor| {
            tensor.deinit();
        }
        self.v.deinit(self.allocator);

        self.* = undefined;
    }

    /// Register a parameter tensor for optimization.
    pub fn addParam(self: *Self, param: *const Tensor) !void {
        const m_tensor = try Tensor.zeros(self.allocator, param.shape);
        try self.m.append(self.allocator, m_tensor);

        const v_tensor = try Tensor.zeros(self.allocator, param.shape);
        try self.v.append(self.allocator, v_tensor);
    }

    /// Perform one optimization step.
    pub fn step(self: *Self, params: []Tensor, grads: []const Tensor) !void {
        if (params.len != grads.len) {
            return error.ParamGradMismatch;
        }
        if (params.len != self.m.items.len) {
            return error.MomentMismatch;
        }

        self.t += 1;
        const t_f: f32 = @floatFromInt(self.t);

        // Bias correction terms
        const bias_correction1 = 1.0 - std.math.pow(f32, self.beta1, t_f);
        const bias_correction2 = 1.0 - std.math.pow(f32, self.beta2, t_f);

        for (params, grads, self.m.items, self.v.items) |*param, grad, *m_tensor, *v_tensor| {
            if (param.data.len != grad.data.len) {
                return error.ShapeMismatch;
            }

            for (param.data, grad.data, m_tensor.data, v_tensor.data) |*p, g, *m, *v| {
                // Update biased first moment
                m.* = self.beta1 * m.* + (1.0 - self.beta1) * g;
                // Update biased second moment
                v.* = self.beta2 * v.* + (1.0 - self.beta2) * g * g;

                // Bias-corrected estimates
                const m_hat = m.* / bias_correction1;
                const v_hat = v.* / bias_correction2;

                // Update parameter
                p.* -= self.lr * m_hat / (@sqrt(v_hat) + self.eps);
            }
        }
    }

    /// Reset optimizer state
    pub fn reset(self: *Self) void {
        self.t = 0;
        for (self.m.items) |*m_tensor| {
            m_tensor.fill(0.0);
        }
        for (self.v.items) |*v_tensor| {
            v_tensor.fill(0.0);
        }
    }
};

// Tests

test "adam basic step" {
    const allocator = std.testing.allocator;

    var optimizer = Adam.init(allocator, 0.001);
    defer optimizer.deinit();

    var param = try Tensor.init(allocator, &[_]usize{3});
    defer param.deinit();
    param.data[0] = 1.0;
    param.data[1] = 2.0;
    param.data[2] = 3.0;

    try optimizer.addParam(&param);

    var grad = try Tensor.ones(allocator, &[_]usize{3});
    defer grad.deinit();

    const initial_0 = param.data[0];

    var params = [_]Tensor{param};
    const grads = [_]Tensor{grad};
    try optimizer.step(&params, &grads);
    param = params[0];

    // Parameters should have decreased (gradient pointing up)
    try std.testing.expect(param.data[0] < initial_0);
}

test "adam converges" {
    const allocator = std.testing.allocator;

    var optimizer = Adam.init(allocator, 0.1);
    defer optimizer.deinit();

    // Simple optimization: minimize (x - 5)^2
    // gradient = 2 * (x - 5)
    var param = try Tensor.zeros(allocator, &[_]usize{1});
    defer param.deinit();

    try optimizer.addParam(&param);

    var grad = try Tensor.zeros(allocator, &[_]usize{1});
    defer grad.deinit();

    // Run several steps
    for (0..100) |_| {
        // Compute gradient: 2 * (x - 5)
        grad.data[0] = 2.0 * (param.data[0] - 5.0);

        var params = [_]Tensor{param};
        const grads = [_]Tensor{grad};
        try optimizer.step(&params, &grads);
        param = params[0];
    }

    // Should be close to 5
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), param.data[0], 0.5);
}

test "adam multiple params" {
    const allocator = std.testing.allocator;

    var optimizer = Adam.init(allocator, 0.01);
    defer optimizer.deinit();

    var param1 = try Tensor.ones(allocator, &[_]usize{2});
    defer param1.deinit();

    var param2 = try Tensor.init(allocator, &[_]usize{2});
    defer param2.deinit();
    param2.fill(10.0);

    try optimizer.addParam(&param1);
    try optimizer.addParam(&param2);

    var grad1 = try Tensor.ones(allocator, &[_]usize{2});
    defer grad1.deinit();

    var grad2 = try Tensor.ones(allocator, &[_]usize{2});
    defer grad2.deinit();
    grad2.fill(-1.0);

    var params = [_]Tensor{ param1, param2 };
    const grads = [_]Tensor{ grad1, grad2 };
    try optimizer.step(&params, &grads);

    // param1 should decrease (positive gradient)
    try std.testing.expect(params[0].data[0] < 1.0);
    // param2 should increase (negative gradient)
    try std.testing.expect(params[1].data[0] > 10.0);
}

test "adam reset" {
    const allocator = std.testing.allocator;

    var optimizer = Adam.init(allocator, 0.001);
    defer optimizer.deinit();

    var param = try Tensor.ones(allocator, &[_]usize{2});
    defer param.deinit();

    try optimizer.addParam(&param);

    var grad = try Tensor.ones(allocator, &[_]usize{2});
    defer grad.deinit();

    // Take some steps
    var params = [_]Tensor{param};
    const grads = [_]Tensor{grad};
    try optimizer.step(&params, &grads);
    try optimizer.step(&params, &grads);

    // Reset
    optimizer.reset();

    // Timestep and moments should be reset
    try std.testing.expectEqual(@as(usize, 0), optimizer.t);
    for (optimizer.m.items[0].data) |m| {
        try std.testing.expectEqual(@as(f32, 0.0), m);
    }
    for (optimizer.v.items[0].data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}
