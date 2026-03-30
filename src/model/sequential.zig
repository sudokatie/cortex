const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Dense = @import("../layers/dense.zig").Dense;
const ReLU = @import("../activations/relu.zig").ReLU;
const Sigmoid = @import("../activations/sigmoid.zig").Sigmoid;
const Tanh = @import("../activations/tanh.zig").Tanh;

/// Layer union - all supported layer types
pub const Layer = union(enum) {
    dense: *Dense,
    relu: *ReLU,
    sigmoid: *Sigmoid,
    tanh: *Tanh,

    /// Forward pass through this layer
    pub fn forward(self: Layer, x: *const Tensor) !Tensor {
        return switch (self) {
            .dense => |d| d.forward(x),
            .relu => |r| r.forward(x),
            .sigmoid => |s| s.forward(x),
            .tanh => |t| t.forward(x),
        };
    }
};

/// Sequential model - chain of layers
pub const Sequential = struct {
    allocator: Allocator,
    layers: std.ArrayListUnmanaged(Layer),
    training: bool,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .layers = .{},
            .training = true,
        };
    }

    pub fn deinit(self: *Self) void {
        self.layers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a dense layer
    pub fn addDense(self: *Self, layer: *Dense) !void {
        try self.layers.append(self.allocator, Layer{ .dense = layer });
    }

    /// Add a ReLU activation
    pub fn addReLU(self: *Self, layer: *ReLU) !void {
        try self.layers.append(self.allocator, Layer{ .relu = layer });
    }

    /// Add a Sigmoid activation
    pub fn addSigmoid(self: *Self, layer: *Sigmoid) !void {
        try self.layers.append(self.allocator, Layer{ .sigmoid = layer });
    }

    /// Add a Tanh activation
    pub fn addTanh(self: *Self, layer: *Tanh) !void {
        try self.layers.append(self.allocator, Layer{ .tanh = layer });
    }

    /// Forward pass through all layers
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        if (self.layers.items.len == 0) {
            return x.clone();
        }

        var current = try self.layers.items[0].forward(x);
        errdefer current.deinit();

        for (self.layers.items[1..]) |layer| {
            const next = try layer.forward(&current);
            current.deinit();
            current = next;
        }

        return current;
    }

    /// Set training mode
    pub fn setTraining(self: *Self, training: bool) void {
        self.training = training;
    }

    /// Get number of layers
    pub fn numLayers(self: *const Self) usize {
        return self.layers.items.len;
    }
};

// Tests

test "sequential empty forward" {
    const allocator = std.testing.allocator;

    var model = Sequential.init(allocator);
    defer model.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 2, 3 });
    defer x.deinit();

    var y = try model.forward(&x);
    defer y.deinit();

    // Empty model returns clone of input
    try std.testing.expectEqualSlices(usize, x.shape, y.shape);
}

test "sequential dense relu dense" {
    const allocator = std.testing.allocator;

    // Create layers
    var dense1 = try Dense.init(allocator, 4, 8);
    defer dense1.deinit();

    var relu1 = ReLU.init(allocator);
    defer relu1.deinit();

    var dense2 = try Dense.init(allocator, 8, 2);
    defer dense2.deinit();

    // Build model
    var model = Sequential.init(allocator);
    defer model.deinit();

    try model.addDense(&dense1);
    try model.addReLU(&relu1);
    try model.addDense(&dense2);

    try std.testing.expectEqual(@as(usize, 3), model.numLayers());

    // Forward pass
    var x = try Tensor.ones(allocator, &[_]usize{ 2, 4 });
    defer x.deinit();

    var y = try model.forward(&x);
    defer y.deinit();

    // Output shape should be [2, 2]
    try std.testing.expectEqual(@as(usize, 2), y.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), y.shape[1]);
}

test "sequential set training" {
    const allocator = std.testing.allocator;

    var model = Sequential.init(allocator);
    defer model.deinit();

    try std.testing.expect(model.training);

    model.setTraining(false);
    try std.testing.expect(!model.training);

    model.setTraining(true);
    try std.testing.expect(model.training);
}

test "sequential with sigmoid" {
    const allocator = std.testing.allocator;

    var dense1 = try Dense.init(allocator, 2, 4);
    defer dense1.deinit();

    var sigmoid1 = Sigmoid.init(allocator);
    defer sigmoid1.deinit();

    var model = Sequential.init(allocator);
    defer model.deinit();

    try model.addDense(&dense1);
    try model.addSigmoid(&sigmoid1);

    var x = try Tensor.ones(allocator, &[_]usize{ 1, 2 });
    defer x.deinit();

    var y = try model.forward(&x);
    defer y.deinit();

    // Output should be [1, 4] with values in (0, 1)
    try std.testing.expectEqual(@as(usize, 1), y.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), y.shape[1]);

    for (y.data) |v| {
        try std.testing.expect(v > 0.0);
        try std.testing.expect(v < 1.0);
    }
}
