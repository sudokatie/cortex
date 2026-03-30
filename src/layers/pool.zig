const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// 2D Max Pooling Layer
/// Input: [batch, channels, height, width]
/// Output: [batch, channels, out_height, out_width]
pub const MaxPool2D = struct {
    allocator: Allocator,
    kernel_size: usize,
    stride: usize,
    /// Cached indices for backward pass
    cached_indices: ?[]usize,
    cached_input_shape: ?[4]usize,

    const Self = @This();

    pub fn init(allocator: Allocator, kernel_size: usize, stride: usize) Self {
        return Self{
            .allocator = allocator,
            .kernel_size = kernel_size,
            .stride = stride,
            .cached_indices = null,
            .cached_input_shape = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.cached_indices) |indices| {
            self.allocator.free(indices);
        }
        self.* = undefined;
    }

    /// Compute output spatial dimension
    pub fn outputSize(self: *const Self, input_size: usize) usize {
        return (input_size - self.kernel_size) / self.stride + 1;
    }

    /// Forward pass
    pub fn forward(self: *Self, input: *const Tensor) !Tensor {
        if (input.shape.len != 4) {
            return error.InvalidInputShape;
        }

        const batch = input.shape[0];
        const channels = input.shape[1];
        const in_h = input.shape[2];
        const in_w = input.shape[3];
        const out_h = self.outputSize(in_h);
        const out_w = self.outputSize(in_w);

        self.cached_input_shape = .{ batch, channels, in_h, in_w };

        // Allocate output and indices
        var output = try Tensor.init(self.allocator, &[_]usize{ batch, channels, out_h, out_w });
        errdefer output.deinit();

        const num_outputs = batch * channels * out_h * out_w;
        if (self.cached_indices) |old| {
            self.allocator.free(old);
        }
        self.cached_indices = try self.allocator.alloc(usize, num_outputs);

        var out_idx: usize = 0;
        for (0..batch) |b| {
            for (0..channels) |c| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        // Find max in kernel window
                        var max_val: f32 = -std.math.inf(f32);
                        var max_idx: usize = 0;

                        for (0..self.kernel_size) |kh| {
                            for (0..self.kernel_size) |kw| {
                                const ih = oh * self.stride + kh;
                                const iw = ow * self.stride + kw;
                                const input_idx = b * channels * in_h * in_w +
                                    c * in_h * in_w +
                                    ih * in_w + iw;

                                if (input.data[input_idx] > max_val) {
                                    max_val = input.data[input_idx];
                                    max_idx = input_idx;
                                }
                            }
                        }

                        output.data[out_idx] = max_val;
                        self.cached_indices.?[out_idx] = max_idx;
                        out_idx += 1;
                    }
                }
            }
        }

        return output;
    }

    /// Backward pass - gradient goes only to max element
    pub fn backward(self: *Self, grad_output: *const Tensor) !Tensor {
        const shape = self.cached_input_shape orelse return error.NoCachedInput;
        const indices = self.cached_indices orelse return error.NoCachedIndices;

        var grad_input = try Tensor.zeros(self.allocator, &[_]usize{ shape[0], shape[1], shape[2], shape[3] });

        for (grad_output.data, indices) |grad, idx| {
            grad_input.data[idx] += grad;
        }

        return grad_input;
    }
};

/// 2D Average Pooling Layer
pub const AvgPool2D = struct {
    allocator: Allocator,
    kernel_size: usize,
    stride: usize,
    cached_input_shape: ?[4]usize,

    const Self = @This();

    pub fn init(allocator: Allocator, kernel_size: usize, stride: usize) Self {
        return Self{
            .allocator = allocator,
            .kernel_size = kernel_size,
            .stride = stride,
            .cached_input_shape = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.* = undefined;
    }

    /// Compute output spatial dimension
    pub fn outputSize(self: *const Self, input_size: usize) usize {
        return (input_size - self.kernel_size) / self.stride + 1;
    }

    /// Forward pass
    pub fn forward(self: *Self, input: *const Tensor) !Tensor {
        if (input.shape.len != 4) {
            return error.InvalidInputShape;
        }

        const batch = input.shape[0];
        const channels = input.shape[1];
        const in_h = input.shape[2];
        const in_w = input.shape[3];
        const out_h = self.outputSize(in_h);
        const out_w = self.outputSize(in_w);

        self.cached_input_shape = .{ batch, channels, in_h, in_w };

        var output = try Tensor.init(self.allocator, &[_]usize{ batch, channels, out_h, out_w });

        const kernel_area: f32 = @floatFromInt(self.kernel_size * self.kernel_size);
        var out_idx: usize = 0;

        for (0..batch) |b| {
            for (0..channels) |c| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        // Compute average in kernel window
                        var sum: f32 = 0.0;

                        for (0..self.kernel_size) |kh| {
                            for (0..self.kernel_size) |kw| {
                                const ih = oh * self.stride + kh;
                                const iw = ow * self.stride + kw;
                                const input_idx = b * channels * in_h * in_w +
                                    c * in_h * in_w +
                                    ih * in_w + iw;
                                sum += input.data[input_idx];
                            }
                        }

                        output.data[out_idx] = sum / kernel_area;
                        out_idx += 1;
                    }
                }
            }
        }

        return output;
    }

    /// Backward pass - gradient distributed evenly
    pub fn backward(self: *Self, grad_output: *const Tensor) !Tensor {
        const shape = self.cached_input_shape orelse return error.NoCachedInput;
        const batch = shape[0];
        const channels = shape[1];
        const in_h = shape[2];
        const in_w = shape[3];
        const out_h = self.outputSize(in_h);
        const out_w = self.outputSize(in_w);

        var grad_input = try Tensor.zeros(self.allocator, &[_]usize{ batch, channels, in_h, in_w });

        const kernel_area: f32 = @floatFromInt(self.kernel_size * self.kernel_size);
        var out_idx: usize = 0;

        for (0..batch) |b| {
            for (0..channels) |c| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        const grad_val = grad_output.data[out_idx] / kernel_area;

                        for (0..self.kernel_size) |kh| {
                            for (0..self.kernel_size) |kw| {
                                const ih = oh * self.stride + kh;
                                const iw = ow * self.stride + kw;
                                const input_idx = b * channels * in_h * in_w +
                                    c * in_h * in_w +
                                    ih * in_w + iw;
                                grad_input.data[input_idx] += grad_val;
                            }
                        }

                        out_idx += 1;
                    }
                }
            }
        }

        return grad_input;
    }
};

// Tests

test "maxpool2d output shape" {
    const allocator = std.testing.allocator;

    var pool = MaxPool2D.init(allocator, 2, 2);
    defer pool.deinit();

    // Input: [1, 1, 4, 4]
    var input = try Tensor.ones(allocator, &[_]usize{ 1, 1, 4, 4 });
    defer input.deinit();

    var output = try pool.forward(&input);
    defer output.deinit();

    // Output: [1, 1, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[3]);
}

test "maxpool2d finds max" {
    const allocator = std.testing.allocator;

    var pool = MaxPool2D.init(allocator, 2, 2);
    defer pool.deinit();

    // Input: [1, 1, 2, 2] with values [1, 2, 3, 4]
    var input = try Tensor.init(allocator, &[_]usize{ 1, 1, 2, 2 });
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;
    input.data[2] = 3.0;
    input.data[3] = 4.0;

    var output = try pool.forward(&input);
    defer output.deinit();

    // Output should be [4.0] (max of the 2x2 window)
    try std.testing.expectEqual(@as(f32, 4.0), output.data[0]);
}

test "maxpool2d backward" {
    const allocator = std.testing.allocator;

    var pool = MaxPool2D.init(allocator, 2, 2);
    defer pool.deinit();

    var input = try Tensor.init(allocator, &[_]usize{ 1, 1, 2, 2 });
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 5.0; // max
    input.data[2] = 3.0;
    input.data[3] = 2.0;

    var output = try pool.forward(&input);
    defer output.deinit();

    var grad_output = try Tensor.init(allocator, &[_]usize{ 1, 1, 1, 1 });
    defer grad_output.deinit();
    grad_output.data[0] = 1.0;

    var grad_input = try pool.backward(&grad_output);
    defer grad_input.deinit();

    // Gradient should go only to max element (index 1)
    try std.testing.expectEqual(@as(f32, 0.0), grad_input.data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), grad_input.data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), grad_input.data[2]);
    try std.testing.expectEqual(@as(f32, 0.0), grad_input.data[3]);
}

test "avgpool2d output shape" {
    const allocator = std.testing.allocator;

    var pool = AvgPool2D.init(allocator, 2, 2);
    defer pool.deinit();

    // Input: [2, 3, 8, 8]
    var input = try Tensor.ones(allocator, &[_]usize{ 2, 3, 8, 8 });
    defer input.deinit();

    var output = try pool.forward(&input);
    defer output.deinit();

    // Output: [2, 3, 4, 4]
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[3]);
}

test "avgpool2d computes average" {
    const allocator = std.testing.allocator;

    var pool = AvgPool2D.init(allocator, 2, 2);
    defer pool.deinit();

    // Input: [1, 1, 2, 2] with values [1, 2, 3, 4]
    var input = try Tensor.init(allocator, &[_]usize{ 1, 1, 2, 2 });
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;
    input.data[2] = 3.0;
    input.data[3] = 4.0;

    var output = try pool.forward(&input);
    defer output.deinit();

    // Output should be [2.5] (average of 1, 2, 3, 4)
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), output.data[0], 0.001);
}

test "avgpool2d backward" {
    const allocator = std.testing.allocator;

    var pool = AvgPool2D.init(allocator, 2, 2);
    defer pool.deinit();

    var input = try Tensor.ones(allocator, &[_]usize{ 1, 1, 2, 2 });
    defer input.deinit();

    var output = try pool.forward(&input);
    defer output.deinit();

    var grad_output = try Tensor.init(allocator, &[_]usize{ 1, 1, 1, 1 });
    defer grad_output.deinit();
    grad_output.data[0] = 4.0;

    var grad_input = try pool.backward(&grad_output);
    defer grad_input.deinit();

    // Gradient should be distributed evenly: 4.0 / 4 = 1.0 to each element
    for (grad_input.data) |g| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), g, 0.001);
    }
}
