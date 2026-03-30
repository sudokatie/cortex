const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const matmul = @import("../tensor/matmul.zig");

/// 2D Convolution Layer using im2col approach
/// Input: [batch, in_channels, height, width]
/// Output: [batch, out_channels, out_height, out_width]
pub const Conv2D = struct {
    allocator: Allocator,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    /// Kernel weights: [out_channels, in_channels * kernel_size * kernel_size]
    weights: Tensor,
    /// Bias: [out_channels]
    bias: Tensor,
    /// Cached im2col result for backward
    cached_col: ?Tensor,
    /// Cached input shape
    cached_input_shape: ?[4]usize,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) !Self {
        const kernel_elements = in_channels * kernel_size * kernel_size;

        var weights = try Tensor.init(allocator, &[_]usize{ out_channels, kernel_elements });
        errdefer weights.deinit();

        // Xavier initialization
        const fan_in = kernel_elements;
        const fan_out = out_channels * kernel_size * kernel_size;
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in + fan_out)));
        var prng = std.Random.DefaultPrng.init(0);
        const rand = prng.random();
        for (weights.data) |*w| {
            w.* = (rand.float(f32) * 2.0 - 1.0) * scale;
        }

        const bias = try Tensor.zeros(allocator, &[_]usize{out_channels});

        return Self{
            .allocator = allocator,
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .weights = weights,
            .bias = bias,
            .cached_col = null,
            .cached_input_shape = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.weights.deinit();
        self.bias.deinit();
        if (self.cached_col) |*cc| {
            cc.deinit();
        }
        self.* = undefined;
    }

    /// Compute output spatial dimensions
    pub fn outputSize(self: *const Self, input_size: usize) usize {
        return (input_size + 2 * self.padding - self.kernel_size) / self.stride + 1;
    }

    /// Extract image patches into columns (im2col)
    fn im2col(self: *const Self, allocator: Allocator, input: *const Tensor) !Tensor {
        const batch = input.shape[0];
        const in_h = input.shape[2];
        const in_w = input.shape[3];
        const out_h = self.outputSize(in_h);
        const out_w = self.outputSize(in_w);

        const col_h = self.in_channels * self.kernel_size * self.kernel_size;
        const col_w = batch * out_h * out_w;

        var col = try Tensor.zeros(allocator, &[_]usize{ col_h, col_w });
        errdefer col.deinit();

        var col_idx: usize = 0;
        for (0..batch) |b| {
            for (0..out_h) |oh| {
                for (0..out_w) |ow| {
                    var row: usize = 0;
                    for (0..self.in_channels) |c| {
                        for (0..self.kernel_size) |kh| {
                            for (0..self.kernel_size) |kw| {
                                const ih = oh * self.stride + kh;
                                const iw = ow * self.stride + kw;

                                // Handle padding
                                if (ih < self.padding or iw < self.padding or
                                    ih >= in_h + self.padding or iw >= in_w + self.padding)
                                {
                                    col.data[row * col_w + col_idx] = 0.0;
                                } else {
                                    const real_ih = ih - self.padding;
                                    const real_iw = iw - self.padding;
                                    const input_idx = b * self.in_channels * in_h * in_w +
                                        c * in_h * in_w +
                                        real_ih * in_w + real_iw;
                                    col.data[row * col_w + col_idx] = input.data[input_idx];
                                }
                                row += 1;
                            }
                        }
                    }
                    col_idx += 1;
                }
            }
        }

        return col;
    }

    /// Forward pass using im2col + matmul
    pub fn forward(self: *Self, input: *const Tensor) !Tensor {
        if (input.shape.len != 4) {
            return error.InvalidInputShape;
        }

        const batch = input.shape[0];
        const in_h = input.shape[2];
        const in_w = input.shape[3];
        const out_h = self.outputSize(in_h);
        const out_w = self.outputSize(in_w);

        // Cache input shape
        self.cached_input_shape = .{ batch, self.in_channels, in_h, in_w };

        // im2col: convert input patches to columns
        if (self.cached_col) |*cc| {
            cc.deinit();
        }
        self.cached_col = try self.im2col(self.allocator, input);

        // Matmul: weights @ col
        // weights: [out_channels, in_channels * k * k]
        // col: [in_channels * k * k, batch * out_h * out_w]
        // result: [out_channels, batch * out_h * out_w]
        var result = try matmul.matmul(self.allocator, &self.weights, &self.cached_col.?);
        errdefer result.deinit();

        // Add bias and reshape to [batch, out_channels, out_h, out_w]
        var output = try Tensor.init(self.allocator, &[_]usize{ batch, self.out_channels, out_h, out_w });

        for (0..batch) |b| {
            for (0..self.out_channels) |oc| {
                for (0..out_h) |oh| {
                    for (0..out_w) |ow| {
                        const col_idx = b * out_h * out_w + oh * out_w + ow;
                        const result_val = result.data[oc * (batch * out_h * out_w) + col_idx];
                        const output_idx = b * self.out_channels * out_h * out_w +
                            oc * out_h * out_w +
                            oh * out_w + ow;
                        output.data[output_idx] = result_val + self.bias.data[oc];
                    }
                }
            }
        }

        result.deinit();
        return output;
    }

    /// Get kernel size (for external use)
    pub fn getKernelSize(self: *const Self) usize {
        return self.kernel_size;
    }
};

// Tests

test "conv2d output shape" {
    const allocator = std.testing.allocator;

    var conv = try Conv2D.init(allocator, 1, 8, 3, 1, 0);
    defer conv.deinit();

    // Input: [1, 1, 8, 8] (batch=1, channels=1, height=8, width=8)
    var input = try Tensor.ones(allocator, &[_]usize{ 1, 1, 8, 8 });
    defer input.deinit();

    var output = try conv.forward(&input);
    defer output.deinit();

    // Output: [1, 8, 6, 6] (kernel 3x3, no padding, stride 1)
    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 8), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 6), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 6), output.shape[3]);
}

test "conv2d with padding" {
    const allocator = std.testing.allocator;

    var conv = try Conv2D.init(allocator, 1, 4, 3, 1, 1);
    defer conv.deinit();

    // Input: [2, 1, 5, 5]
    var input = try Tensor.ones(allocator, &[_]usize{ 2, 1, 5, 5 });
    defer input.deinit();

    var output = try conv.forward(&input);
    defer output.deinit();

    // With padding=1, same spatial size: [2, 4, 5, 5]
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 5), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 5), output.shape[3]);
}

test "conv2d with stride" {
    const allocator = std.testing.allocator;

    var conv = try Conv2D.init(allocator, 3, 16, 3, 2, 1);
    defer conv.deinit();

    // Input: [1, 3, 32, 32]
    var input = try Tensor.ones(allocator, &[_]usize{ 1, 3, 32, 32 });
    defer input.deinit();

    var output = try conv.forward(&input);
    defer output.deinit();

    // With stride=2: (32 + 2*1 - 3) / 2 + 1 = 16
    try std.testing.expectEqual(@as(usize, 1), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 16), output.shape[1]);
    try std.testing.expectEqual(@as(usize, 16), output.shape[2]);
    try std.testing.expectEqual(@as(usize, 16), output.shape[3]);
}

test "conv2d output size calculation" {
    const allocator = std.testing.allocator;

    var conv = try Conv2D.init(allocator, 1, 1, 5, 2, 2);
    defer conv.deinit();

    // (10 + 2*2 - 5) / 2 + 1 = 5
    try std.testing.expectEqual(@as(usize, 5), conv.outputSize(10));

    // (28 + 2*2 - 5) / 2 + 1 = 14
    try std.testing.expectEqual(@as(usize, 14), conv.outputSize(28));
}
