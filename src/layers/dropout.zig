const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Dropout layer.
/// During training, randomly zeroes elements with probability p.
/// During inference, returns input unchanged.
/// The remaining elements are scaled by 1/(1-p) to maintain expected values.
pub const Dropout = struct {
    /// Dropout probability (fraction of elements to zero)
    p: f32,
    /// Whether the layer is in training mode
    training: bool,
    /// Cached mask for backward pass
    cached_mask: ?Tensor,
    /// Random number generator
    rng: std.Random.DefaultPrng,
    /// Allocator
    allocator: Allocator,

    const Self = @This();

    /// Create a new dropout layer.
    /// p: probability of dropping each element (default 0.5)
    pub fn init(allocator: Allocator, p: f32) Self {
        return Self{
            .p = p,
            .training = true,
            .cached_mask = null,
            .rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp())),
            .allocator = allocator,
        };
    }

    /// Create dropout with seed for reproducibility.
    pub fn initWithSeed(allocator: Allocator, p: f32, seed: u64) Self {
        return Self{
            .p = p,
            .training = true,
            .cached_mask = null,
            .rng = std.Random.DefaultPrng.init(seed),
            .allocator = allocator,
        };
    }

    /// Free all memory.
    pub fn deinit(self: *Self) void {
        if (self.cached_mask) |*mask| {
            mask.deinit();
        }
        self.* = undefined;
    }

    /// Set training mode.
    pub fn train(self: *Self, mode: bool) void {
        self.training = mode;
    }

    /// Forward pass.
    /// During training: randomly zeros elements and scales remaining by 1/(1-p).
    /// During inference: returns input unchanged.
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        // Clear cached mask
        if (self.cached_mask) |*mask| {
            mask.deinit();
            self.cached_mask = null;
        }

        if (!self.training or self.p == 0.0) {
            // Inference mode or no dropout - just clone input
            return x.clone();
        }

        if (self.p >= 1.0) {
            // Drop everything
            return Tensor.zeros(self.allocator, x.shape);
        }

        var output = try Tensor.init(self.allocator, x.shape);
        errdefer output.deinit();

        var mask = try Tensor.init(self.allocator, x.shape);
        errdefer mask.deinit();

        const scale = 1.0 / (1.0 - self.p);
        const rand = self.rng.random();

        for (0..x.data.len) |i| {
            if (rand.float(f32) >= self.p) {
                // Keep this element, scale it
                output.data[i] = x.data[i] * scale;
                mask.data[i] = scale;
            } else {
                // Drop this element
                output.data[i] = 0.0;
                mask.data[i] = 0.0;
            }
        }

        self.cached_mask = mask;

        return output;
    }

    /// Backward pass.
    /// Applies the same mask used in forward pass.
    pub fn backward(self: *Self, grad_output: *const Tensor) !Tensor {
        if (!self.training or self.p == 0.0) {
            return grad_output.clone();
        }

        const mask = &(self.cached_mask orelse return error.NoCachedMask);

        var grad_input = try Tensor.init(self.allocator, grad_output.shape);
        errdefer grad_input.deinit();

        for (0..grad_output.data.len) |i| {
            grad_input.data[i] = grad_output.data[i] * mask.data[i];
        }

        return grad_input;
    }
};

// Tests

test "dropout forward shape" {
    const allocator = std.testing.allocator;

    var dropout = Dropout.initWithSeed(allocator, 0.5, 42);
    defer dropout.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 2, 4 });
    defer x.deinit();

    var y = try dropout.forward(&x);
    defer y.deinit();

    try std.testing.expectEqual(@as(usize, 2), y.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), y.shape[1]);
}

test "dropout zeros some elements" {
    const allocator = std.testing.allocator;

    var dropout = Dropout.initWithSeed(allocator, 0.5, 42);
    defer dropout.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 100, 100 });
    defer x.deinit();

    var y = try dropout.forward(&x);
    defer y.deinit();

    // Count zeros - should be roughly half with p=0.5
    var zero_count: usize = 0;
    for (y.data) |val| {
        if (val == 0.0) zero_count += 1;
    }

    const total = x.data.len;
    const zero_ratio = @as(f32, @floatFromInt(zero_count)) / @as(f32, @floatFromInt(total));

    // With 10000 elements and p=0.5, we expect roughly 50% zeros
    // Allow some variance (40% to 60%)
    try std.testing.expect(zero_ratio > 0.4);
    try std.testing.expect(zero_ratio < 0.6);
}

test "dropout scales non-zero elements" {
    const allocator = std.testing.allocator;

    var dropout = Dropout.initWithSeed(allocator, 0.5, 42);
    defer dropout.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 10, 10 });
    defer x.deinit();

    var y = try dropout.forward(&x);
    defer y.deinit();

    // Non-zero elements should be scaled by 1/(1-p) = 2
    for (y.data) |val| {
        if (val != 0.0) {
            try std.testing.expectApproxEqAbs(@as(f32, 2.0), val, 0.001);
        }
    }
}

test "dropout inference mode passes through" {
    const allocator = std.testing.allocator;

    var dropout = Dropout.initWithSeed(allocator, 0.5, 42);
    defer dropout.deinit();

    // Switch to eval mode
    dropout.train(false);

    var x = try Tensor.ones(allocator, &[_]usize{ 2, 4 });
    defer x.deinit();

    var y = try dropout.forward(&x);
    defer y.deinit();

    // All elements should be unchanged
    for (y.data) |val| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), val, 0.001);
    }
}

test "dropout zero probability passes through" {
    const allocator = std.testing.allocator;

    var dropout = Dropout.initWithSeed(allocator, 0.0, 42);
    defer dropout.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 2, 4 });
    defer x.deinit();

    var y = try dropout.forward(&x);
    defer y.deinit();

    // All elements should be unchanged
    for (y.data) |val| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), val, 0.001);
    }
}

test "dropout backward applies mask" {
    const allocator = std.testing.allocator;

    var dropout = Dropout.initWithSeed(allocator, 0.5, 42);
    defer dropout.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 4, 4 });
    defer x.deinit();

    // Forward pass creates the mask
    var y = try dropout.forward(&x);
    defer y.deinit();

    var grad_out = try Tensor.ones(allocator, &[_]usize{ 4, 4 });
    defer grad_out.deinit();

    var grad_in = try dropout.backward(&grad_out);
    defer grad_in.deinit();

    // Gradient should match output pattern (same zeros)
    for (0..y.data.len) |i| {
        if (y.data[i] == 0.0) {
            try std.testing.expectApproxEqAbs(@as(f32, 0.0), grad_in.data[i], 0.001);
        } else {
            try std.testing.expectApproxEqAbs(@as(f32, 2.0), grad_in.data[i], 0.001);
        }
    }
}
