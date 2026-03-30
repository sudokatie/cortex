const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const softmax_mod = @import("../activations/softmax.zig");

/// Cross-entropy loss with softmax: -sum(y_true * log(softmax(logits)))
/// Combined softmax + cross-entropy for numerical stability.
pub const CrossEntropyLoss = struct {
    allocator: Allocator,
    cached_softmax: ?Tensor,
    cached_target: ?Tensor,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .cached_softmax = null,
            .cached_target = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.cached_softmax) |*cs| {
            cs.deinit();
        }
        if (self.cached_target) |*ct| {
            ct.deinit();
        }
        self.* = undefined;
    }

    /// Forward pass: compute loss
    /// logits: [batch, classes] - raw model outputs (not softmaxed)
    /// target: [batch, classes] - one-hot encoded targets
    /// Returns: scalar loss (sum over batch, mean over classes)
    pub fn forward(self: *Self, logits: *const Tensor, target: *const Tensor) !f32 {
        if (logits.shape.len != 2 or target.shape.len != 2) {
            return error.InvalidShape;
        }
        if (!logits.sameShape(target)) {
            return error.ShapeMismatch;
        }

        const batch_size = logits.shape[0];
        const num_classes = logits.shape[1];

        // Compute softmax
        var probs = try softmax_mod.softmax(self.allocator, logits);
        errdefer probs.deinit();

        // Cache for backward
        if (self.cached_softmax) |*cs| {
            cs.deinit();
        }
        self.cached_softmax = probs;

        if (self.cached_target) |*ct| {
            ct.deinit();
        }
        self.cached_target = try target.clone();

        // Compute cross-entropy: -sum(y_true * log(probs))
        var total_loss: f32 = 0.0;
        const eps: f32 = 1e-7; // Small epsilon to avoid log(0)

        for (0..batch_size) |b| {
            for (0..num_classes) |c| {
                const idx = b * num_classes + c;
                const t = target.data[idx];
                const p = probs.data[idx];
                if (t > 0.0) {
                    total_loss -= t * @log(@max(p, eps));
                }
            }
        }

        // Return mean loss over batch
        return total_loss / @as(f32, @floatFromInt(batch_size));
    }

    /// Backward pass: gradient w.r.t. logits
    /// The combined softmax + cross-entropy gradient is simply: softmax - target
    pub fn backward(self: *Self) !Tensor {
        const probs = &(self.cached_softmax orelse return error.NoCachedSoftmax);
        const target = &(self.cached_target orelse return error.NoCachedTarget);

        const batch_size = probs.shape[0];

        const grad = try Tensor.init(self.allocator, probs.shape);
        for (grad.data, probs.data, target.data) |*g, p, t| {
            g.* = (p - t) / @as(f32, @floatFromInt(batch_size));
        }

        return grad;
    }
};

/// Functional cross-entropy loss (no caching)
/// Returns (loss, softmax_probs)
pub fn crossEntropyLoss(allocator: Allocator, logits: *const Tensor, target: *const Tensor) !struct { loss: f32, probs: Tensor } {
    if (logits.shape.len != 2 or target.shape.len != 2) {
        return error.InvalidShape;
    }

    const batch_size = logits.shape[0];
    const num_classes = logits.shape[1];

    var probs = try softmax_mod.softmax(allocator, logits);
    errdefer probs.deinit();

    var total_loss: f32 = 0.0;
    const eps: f32 = 1e-7;

    for (0..batch_size) |b| {
        for (0..num_classes) |c| {
            const idx = b * num_classes + c;
            const t = target.data[idx];
            const p = probs.data[idx];
            if (t > 0.0) {
                total_loss -= t * @log(@max(p, eps));
            }
        }
    }

    return .{
        .loss = total_loss / @as(f32, @floatFromInt(batch_size)),
        .probs = probs,
    };
}

// Tests

test "cross entropy with one-hot" {
    const allocator = std.testing.allocator;

    // Logits: [2, 3] - two samples, three classes
    var logits = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer logits.deinit();
    // Sample 0: class 2 is highest (correct)
    logits.data[0] = 1.0;
    logits.data[1] = 2.0;
    logits.data[2] = 3.0;
    // Sample 1: class 0 is highest (correct)
    logits.data[3] = 3.0;
    logits.data[4] = 2.0;
    logits.data[5] = 1.0;

    // Targets: one-hot
    var target = try Tensor.zeros(allocator, &[_]usize{ 2, 3 });
    defer target.deinit();
    target.data[2] = 1.0; // Sample 0: class 2
    target.data[3] = 1.0; // Sample 1: class 0

    var loss_fn = CrossEntropyLoss.init(allocator);
    defer loss_fn.deinit();

    const loss = try loss_fn.forward(&logits, &target);

    // Loss should be relatively small since predictions match targets
    // -log(softmax(3)) for both samples, which is about -log(0.665) ≈ 0.41
    try std.testing.expect(loss > 0.0);
    try std.testing.expect(loss < 1.0); // Should be smallish
}

test "cross entropy gradient shape" {
    const allocator = std.testing.allocator;

    var logits = try Tensor.init(allocator, &[_]usize{ 4, 5 });
    defer logits.deinit();
    for (logits.data) |*v| v.* = 0.0;

    var target = try Tensor.zeros(allocator, &[_]usize{ 4, 5 });
    defer target.deinit();
    for (0..4) |b| {
        target.data[b * 5 + b % 5] = 1.0;
    }

    var loss_fn = CrossEntropyLoss.init(allocator);
    defer loss_fn.deinit();

    _ = try loss_fn.forward(&logits, &target);

    var grad = try loss_fn.backward();
    defer grad.deinit();

    // Gradient should have same shape as logits
    try std.testing.expectEqual(@as(usize, 4), grad.shape[0]);
    try std.testing.expectEqual(@as(usize, 5), grad.shape[1]);
}

test "cross entropy gradient values" {
    const allocator = std.testing.allocator;

    // Simple case: 1 sample, 2 classes
    var logits = try Tensor.init(allocator, &[_]usize{ 1, 2 });
    defer logits.deinit();
    logits.data[0] = 0.0;
    logits.data[1] = 0.0; // Equal logits = softmax 0.5, 0.5

    var target = try Tensor.zeros(allocator, &[_]usize{ 1, 2 });
    defer target.deinit();
    target.data[0] = 1.0; // One-hot: class 0

    var loss_fn = CrossEntropyLoss.init(allocator);
    defer loss_fn.deinit();

    _ = try loss_fn.forward(&logits, &target);

    var grad = try loss_fn.backward();
    defer grad.deinit();

    // grad = (softmax - target) / batch_size
    // softmax = [0.5, 0.5], target = [1, 0]
    // grad = [0.5 - 1, 0.5 - 0] / 1 = [-0.5, 0.5]
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), grad.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad.data[1], 0.001);
}

test "cross entropy high loss for wrong prediction" {
    const allocator = std.testing.allocator;

    // Logits strongly predict wrong class
    var logits = try Tensor.init(allocator, &[_]usize{ 1, 3 });
    defer logits.deinit();
    logits.data[0] = 10.0; // Strong prediction for class 0
    logits.data[1] = 0.0;
    logits.data[2] = 0.0;

    // But target is class 2
    var target = try Tensor.zeros(allocator, &[_]usize{ 1, 3 });
    defer target.deinit();
    target.data[2] = 1.0;

    var loss_fn = CrossEntropyLoss.init(allocator);
    defer loss_fn.deinit();

    const loss = try loss_fn.forward(&logits, &target);

    // Loss should be high because softmax(class 2) is very small
    try std.testing.expect(loss > 5.0);
}
