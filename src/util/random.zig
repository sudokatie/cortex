const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// PCG random number generator for reproducibility
pub const Random = struct {
    prng: std.Random.DefaultPrng,

    const Self = @This();

    pub fn init(seed: u64) Self {
        return Self{
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    /// Generate uniform random value in [0, 1)
    pub fn uniform(self: *Self) f32 {
        return self.prng.random().float(f32);
    }

    /// Generate uniform random value in [low, high)
    pub fn uniformRange(self: *Self, low: f32, high: f32) f32 {
        return low + (high - low) * self.uniform();
    }

    /// Generate normal (Gaussian) random value using Box-Muller transform
    pub fn normal(self: *Self, mean: f32, std_dev: f32) f32 {
        // Box-Muller transform
        const rand1 = @max(self.uniform(), 1e-10); // Avoid log(0)
        const rand2 = self.uniform();
        const z = @sqrt(-2.0 * @log(rand1)) * @cos(2.0 * std.math.pi * rand2);
        return mean + std_dev * z;
    }

    /// Fill tensor with uniform random values in [low, high)
    pub fn fillUniform(self: *Self, tensor: *Tensor, low: f32, high: f32) void {
        for (tensor.data) |*v| {
            v.* = self.uniformRange(low, high);
        }
    }

    /// Fill tensor with normal random values
    pub fn fillNormal(self: *Self, tensor: *Tensor, mean: f32, std_dev: f32) void {
        for (tensor.data) |*v| {
            v.* = self.normal(mean, std_dev);
        }
    }
};

/// Weight initialization strategies
pub const WeightInit = struct {
    /// Xavier/Glorot initialization
    /// scale = sqrt(2 / (fan_in + fan_out))
    pub fn xavier(allocator: Allocator, rng: *Random, shape: []const usize) !Tensor {
        if (shape.len < 2) {
            return error.InvalidShape;
        }

        const fan_in = shape[0];
        const fan_out = shape[1];
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in + fan_out)));

        var tensor = try Tensor.init(allocator, shape);
        rng.fillNormal(&tensor, 0.0, scale);
        return tensor;
    }

    /// Kaiming/He initialization (for ReLU networks)
    /// scale = sqrt(2 / fan_in)
    pub fn kaiming(allocator: Allocator, rng: *Random, shape: []const usize) !Tensor {
        if (shape.len < 2) {
            return error.InvalidShape;
        }

        const fan_in = shape[0];
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in)));

        var tensor = try Tensor.init(allocator, shape);
        rng.fillNormal(&tensor, 0.0, scale);
        return tensor;
    }

    /// LeCun initialization
    /// scale = sqrt(1 / fan_in)
    pub fn lecun(allocator: Allocator, rng: *Random, shape: []const usize) !Tensor {
        if (shape.len < 2) {
            return error.InvalidShape;
        }

        const fan_in = shape[0];
        const scale = @sqrt(1.0 / @as(f32, @floatFromInt(fan_in)));

        var tensor = try Tensor.init(allocator, shape);
        rng.fillNormal(&tensor, 0.0, scale);
        return tensor;
    }

    /// Uniform initialization in [-limit, limit]
    pub fn uniformInit(allocator: Allocator, rng: *Random, shape: []const usize, limit: f32) !Tensor {
        var tensor = try Tensor.init(allocator, shape);
        rng.fillUniform(&tensor, -limit, limit);
        return tensor;
    }
};

// Tests

test "random uniform range" {
    var rng = Random.init(42);

    for (0..100) |_| {
        const v = rng.uniform();
        try std.testing.expect(v >= 0.0);
        try std.testing.expect(v < 1.0);
    }
}

test "random uniform range custom" {
    var rng = Random.init(123);

    for (0..100) |_| {
        const v = rng.uniformRange(-5.0, 5.0);
        try std.testing.expect(v >= -5.0);
        try std.testing.expect(v < 5.0);
    }
}

test "random normal distribution" {
    var rng = Random.init(456);

    var sum: f32 = 0.0;
    var sum_sq: f32 = 0.0;
    const n: usize = 10000;

    for (0..n) |_| {
        const v = rng.normal(0.0, 1.0);
        sum += v;
        sum_sq += v * v;
    }

    const mean = sum / @as(f32, @floatFromInt(n));
    const variance = sum_sq / @as(f32, @floatFromInt(n)) - mean * mean;
    const std_dev = @sqrt(variance);

    // Mean should be close to 0
    try std.testing.expect(@abs(mean) < 0.1);
    // Std dev should be close to 1
    try std.testing.expect(@abs(std_dev - 1.0) < 0.1);
}

test "fill uniform" {
    const allocator = std.testing.allocator;
    var rng = Random.init(789);

    var tensor = try Tensor.init(allocator, &[_]usize{100});
    defer tensor.deinit();

    rng.fillUniform(&tensor, -1.0, 1.0);

    for (tensor.data) |v| {
        try std.testing.expect(v >= -1.0);
        try std.testing.expect(v < 1.0);
    }
}

test "xavier init" {
    const allocator = std.testing.allocator;
    var rng = Random.init(101);

    var weights = try WeightInit.xavier(allocator, &rng, &[_]usize{ 100, 50 });
    defer weights.deinit();

    // Expected std_dev = sqrt(2 / (100 + 50)) = sqrt(2/150) ≈ 0.115
    var sum: f32 = 0.0;
    var sum_sq: f32 = 0.0;
    for (weights.data) |v| {
        sum += v;
        sum_sq += v * v;
    }
    const n = @as(f32, @floatFromInt(weights.data.len));
    const mean = sum / n;
    const variance = sum_sq / n - mean * mean;
    const std_dev = @sqrt(variance);

    // Should be roughly 0.115
    try std.testing.expect(std_dev > 0.05);
    try std.testing.expect(std_dev < 0.25);
}

test "kaiming init" {
    const allocator = std.testing.allocator;
    var rng = Random.init(202);

    var weights = try WeightInit.kaiming(allocator, &rng, &[_]usize{ 100, 50 });
    defer weights.deinit();

    // Expected std_dev = sqrt(2 / 100) = sqrt(0.02) ≈ 0.141
    var sum_sq: f32 = 0.0;
    for (weights.data) |v| {
        sum_sq += v * v;
    }
    const n = @as(f32, @floatFromInt(weights.data.len));
    const std_dev = @sqrt(sum_sq / n);

    // Should be roughly 0.141
    try std.testing.expect(std_dev > 0.07);
    try std.testing.expect(std_dev < 0.28);
}
