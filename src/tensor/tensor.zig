const std = @import("std");
const Allocator = std.mem.Allocator;

/// Multi-dimensional array for numerical computation.
pub const Tensor = struct {
    /// Shape of the tensor (e.g., [batch, channels, height, width])
    shape: []usize,
    /// Strides for indexing (bytes to skip for each dimension)
    strides: []usize,
    /// Contiguous data storage
    data: []f32,
    /// Allocator for memory management
    allocator: Allocator,

    const Self = @This();

    /// Create a new tensor with the given shape.
    /// Data is uninitialized.
    pub fn init(allocator: Allocator, shape: []const usize) !Self {
        const num_dims = shape.len;
        if (num_dims == 0) {
            return error.InvalidShape;
        }

        // Calculate total size
        var total_size: usize = 1;
        for (shape) |dim| {
            if (dim == 0) {
                return error.InvalidShape;
            }
            total_size *= dim;
        }

        // Allocate shape and strides
        const owned_shape = try allocator.alloc(usize, num_dims);
        @memcpy(owned_shape, shape);

        const strides = try allocator.alloc(usize, num_dims);
        var stride: usize = 1;
        var i: usize = num_dims;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= shape[i];
        }

        // Allocate data
        const data = try allocator.alloc(f32, total_size);

        return Self{
            .shape = owned_shape,
            .strides = strides,
            .data = data,
            .allocator = allocator,
        };
    }

    /// Free all memory.
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
        self.allocator.free(self.strides);
        self.allocator.free(self.shape);
        self.* = undefined;
    }

    /// Get the number of dimensions.
    pub fn ndim(self: *const Self) usize {
        return self.shape.len;
    }

    /// Get the total number of elements.
    pub fn size(self: *const Self) usize {
        var s: usize = 1;
        for (self.shape) |dim| {
            s *= dim;
        }
        return s;
    }

    /// Convert multi-dimensional indices to flat index.
    fn flatIndex(self: *const Self, indices: []const usize) usize {
        var idx: usize = 0;
        for (indices, self.strides) |i, s| {
            idx += i * s;
        }
        return idx;
    }

    /// Get element at indices.
    pub fn at(self: *const Self, indices: []const usize) f32 {
        const idx = self.flatIndex(indices);
        return self.data[idx];
    }

    /// Set element at indices.
    pub fn set(self: *Self, indices: []const usize, value: f32) void {
        const idx = self.flatIndex(indices);
        self.data[idx] = value;
    }

    /// Fill all elements with a value.
    pub fn fill(self: *Self, value: f32) void {
        @memset(self.data, value);
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(allocator: Allocator, shape: []const usize) !Self {
        var t = try Self.init(allocator, shape);
        t.fill(0.0);
        return t;
    }

    /// Create a tensor filled with ones.
    pub fn ones(allocator: Allocator, shape: []const usize) !Self {
        var t = try Self.init(allocator, shape);
        t.fill(1.0);
        return t;
    }

    /// Create a deep copy.
    pub fn clone(self: *const Self) !Self {
        const new = try Self.init(self.allocator, self.shape);
        @memcpy(new.data, self.data);
        return new;
    }

    /// Reshape to new shape (must have same total size).
    pub fn reshape(self: *const Self, new_shape: []const usize) !Self {
        // Calculate new size
        var new_size: usize = 1;
        for (new_shape) |dim| {
            new_size *= dim;
        }

        if (new_size != self.size()) {
            return error.IncompatibleShape;
        }

        // Create new tensor sharing data (for now, copy)
        const new = try Self.init(self.allocator, new_shape);
        @memcpy(new.data, self.data);
        return new;
    }

    /// Check if two tensors have the same shape.
    pub fn sameShape(self: *const Self, other: *const Self) bool {
        if (self.shape.len != other.shape.len) return false;
        for (self.shape, other.shape) |a, b| {
            if (a != b) return false;
        }
        return true;
    }

    /// Print tensor for debugging.
    pub fn print(self: *const Self, writer: anytype) !void {
        try writer.print("Tensor(shape=[", .{});
        for (self.shape, 0..) |dim, i| {
            if (i > 0) try writer.print(", ", .{});
            try writer.print("{}", .{dim});
        }
        try writer.print("], data=[", .{});

        const max_print: usize = 10;
        const len = @min(self.data.len, max_print);
        for (0..len) |i| {
            if (i > 0) try writer.print(", ", .{});
            try writer.print("{d:.4}", .{self.data[i]});
        }
        if (self.data.len > max_print) {
            try writer.print(", ...", .{});
        }
        try writer.print("])\n", .{});
    }
};

test "tensor creation" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 2), t.ndim());
    try std.testing.expectEqual(@as(usize, 6), t.size());
    try std.testing.expectEqual(@as(usize, 2), t.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), t.shape[1]);
}

test "tensor get/set" {
    const allocator = std.testing.allocator;

    var t = try Tensor.zeros(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();

    t.set(&[_]usize{ 0, 0 }, 1.0);
    t.set(&[_]usize{ 1, 2 }, 5.0);

    try std.testing.expectEqual(@as(f32, 1.0), t.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 5.0), t.at(&[_]usize{ 1, 2 }));
    try std.testing.expectEqual(@as(f32, 0.0), t.at(&[_]usize{ 0, 1 }));
}

test "tensor zeros and ones" {
    const allocator = std.testing.allocator;

    var z = try Tensor.zeros(allocator, &[_]usize{4});
    defer z.deinit();
    for (z.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    var o = try Tensor.ones(allocator, &[_]usize{4});
    defer o.deinit();
    for (o.data) |v| {
        try std.testing.expectEqual(@as(f32, 1.0), v);
    }
}

test "tensor clone" {
    const allocator = std.testing.allocator;

    var t = try Tensor.zeros(allocator, &[_]usize{ 2, 2 });
    defer t.deinit();
    t.set(&[_]usize{ 0, 0 }, 42.0);

    var c = try t.clone();
    defer c.deinit();

    try std.testing.expectEqual(@as(f32, 42.0), c.at(&[_]usize{ 0, 0 }));

    // Modify original, clone should be independent
    t.set(&[_]usize{ 0, 0 }, 0.0);
    try std.testing.expectEqual(@as(f32, 42.0), c.at(&[_]usize{ 0, 0 }));
}

test "tensor reshape" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();
    for (t.data, 0..) |*v, i| {
        v.* = @floatFromInt(i);
    }

    var r = try t.reshape(&[_]usize{ 3, 2 });
    defer r.deinit();

    try std.testing.expectEqual(@as(usize, 3), r.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), r.shape[1]);
    try std.testing.expectEqual(@as(f32, 0.0), r.data[0]);
    try std.testing.expectEqual(@as(f32, 5.0), r.data[5]);
}

test "tensor reshape incompatible" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();

    const result = t.reshape(&[_]usize{ 2, 2 });
    try std.testing.expectError(error.IncompatibleShape, result);
}

test "tensor same shape" {
    const allocator = std.testing.allocator;

    var a = try Tensor.zeros(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();

    var b = try Tensor.zeros(allocator, &[_]usize{ 2, 3 });
    defer b.deinit();

    var c = try Tensor.zeros(allocator, &[_]usize{ 3, 2 });
    defer c.deinit();

    try std.testing.expect(a.sameShape(&b));
    try std.testing.expect(!a.sameShape(&c));
}
