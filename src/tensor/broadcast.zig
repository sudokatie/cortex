const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

/// Maximum number of dimensions supported
const MAX_DIMS: usize = 8;

/// Check if two shapes can be broadcast together.
/// Broadcasting rules:
/// - Shapes are aligned from the right
/// - Dimension sizes must be equal, or one must be 1
/// - Missing dimensions are treated as 1
pub fn canBroadcast(shape_a: []const usize, shape_b: []const usize) bool {
    const max_dims = @max(shape_a.len, shape_b.len);

    var i: usize = 0;
    while (i < max_dims) : (i += 1) {
        const dim_a = if (i < shape_a.len) shape_a[shape_a.len - 1 - i] else 1;
        const dim_b = if (i < shape_b.len) shape_b[shape_b.len - 1 - i] else 1;

        if (dim_a != dim_b and dim_a != 1 and dim_b != 1) {
            return false;
        }
    }
    return true;
}

/// Compute the resulting shape when broadcasting two shapes together.
/// Returns error if shapes are incompatible.
pub fn broadcastShape(allocator: Allocator, shape_a: []const usize, shape_b: []const usize) ![]usize {
    if (!canBroadcast(shape_a, shape_b)) {
        return error.IncompatibleShapes;
    }

    const max_dims = @max(shape_a.len, shape_b.len);
    const result = try allocator.alloc(usize, max_dims);
    errdefer allocator.free(result);

    var i: usize = 0;
    while (i < max_dims) : (i += 1) {
        const dim_a = if (i < shape_a.len) shape_a[shape_a.len - 1 - i] else 1;
        const dim_b = if (i < shape_b.len) shape_b[shape_b.len - 1 - i] else 1;
        result[max_dims - 1 - i] = @max(dim_a, dim_b);
    }

    return result;
}

/// Expand a tensor to a target shape via broadcasting.
/// The tensor data is copied and repeated as needed.
pub fn broadcast(allocator: Allocator, tensor: *const Tensor, target_shape: []const usize) !Tensor {
    // Verify broadcast is valid
    if (!canBroadcast(tensor.shape, target_shape)) {
        return error.IncompatibleShapes;
    }

    // Create result tensor
    var result = try Tensor.init(allocator, target_shape);
    errdefer result.deinit();

    // Calculate total elements in result
    var total_size: usize = 1;
    for (target_shape) |dim| {
        total_size *= dim;
    }

    // Fill result by mapping each index back to source
    var indices: [MAX_DIMS]usize = undefined;
    const num_dims = target_shape.len;

    var flat_idx: usize = 0;
    while (flat_idx < total_size) : (flat_idx += 1) {
        // Convert flat index to multi-dimensional indices
        var temp = flat_idx;
        var dim_idx: usize = num_dims;
        while (dim_idx > 0) {
            dim_idx -= 1;
            indices[dim_idx] = temp % target_shape[dim_idx];
            temp /= target_shape[dim_idx];
        }

        // Map to source indices (handle broadcasting)
        var src_indices: [MAX_DIMS]usize = undefined;
        const src_dims = tensor.shape.len;
        const offset = num_dims - src_dims;

        for (0..src_dims) |d| {
            const target_idx = d + offset;
            if (tensor.shape[d] == 1) {
                src_indices[d] = 0; // Broadcast: always use index 0
            } else {
                src_indices[d] = indices[target_idx];
            }
        }

        // Get value from source and set in result
        const value = tensor.at(src_indices[0..src_dims]);
        result.data[flat_idx] = value;
    }

    return result;
}

/// Broadcast two tensors to a common shape.
/// Returns both tensors expanded to the same shape.
pub fn broadcastPair(allocator: Allocator, a: *const Tensor, b: *const Tensor) !struct { a: Tensor, b: Tensor } {
    const result_shape = try broadcastShape(allocator, a.shape, b.shape);
    defer allocator.free(result_shape);

    var broadcast_a = try broadcast(allocator, a, result_shape);
    errdefer broadcast_a.deinit();

    const broadcast_b = try broadcast(allocator, b, result_shape);

    return .{ .a = broadcast_a, .b = broadcast_b };
}

// Tests

test "canBroadcast basic" {
    // Same shape
    try std.testing.expect(canBroadcast(&[_]usize{ 2, 3 }, &[_]usize{ 2, 3 }));

    // One dimension is 1
    try std.testing.expect(canBroadcast(&[_]usize{ 3, 1 }, &[_]usize{ 1, 4 }));
    try std.testing.expect(canBroadcast(&[_]usize{ 3, 1 }, &[_]usize{ 3, 4 }));

    // Different number of dimensions
    try std.testing.expect(canBroadcast(&[_]usize{ 2, 3, 4 }, &[_]usize{4}));
    try std.testing.expect(canBroadcast(&[_]usize{4}, &[_]usize{ 2, 3, 4 }));

    // Incompatible
    try std.testing.expect(!canBroadcast(&[_]usize{ 2, 3 }, &[_]usize{ 3, 4 }));
    try std.testing.expect(!canBroadcast(&[_]usize{2}, &[_]usize{3}));
}

test "broadcastShape" {
    const allocator = std.testing.allocator;

    // [3,1] + [1,4] -> [3,4]
    {
        const shape = try broadcastShape(allocator, &[_]usize{ 3, 1 }, &[_]usize{ 1, 4 });
        defer allocator.free(shape);
        try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 4 }, shape);
    }

    // [2,3,4] + [4] -> [2,3,4]
    {
        const shape = try broadcastShape(allocator, &[_]usize{ 2, 3, 4 }, &[_]usize{4});
        defer allocator.free(shape);
        try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 4 }, shape);
    }

    // [1,5] + [3,1] -> [3,5]
    {
        const shape = try broadcastShape(allocator, &[_]usize{ 1, 5 }, &[_]usize{ 3, 1 });
        defer allocator.free(shape);
        try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 5 }, shape);
    }

    // Incompatible should fail
    {
        const result = broadcastShape(allocator, &[_]usize{ 2, 3 }, &[_]usize{ 3, 4 });
        try std.testing.expectError(error.IncompatibleShapes, result);
    }
}

test "broadcast scalar to matrix" {
    const allocator = std.testing.allocator;

    // Scalar [1] broadcast to [2,3]
    var scalar = try Tensor.init(allocator, &[_]usize{1});
    defer scalar.deinit();
    scalar.set(&[_]usize{0}, 5.0);

    var result = try broadcast(allocator, &scalar, &[_]usize{ 2, 3 });
    defer result.deinit();

    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, result.shape);
    for (result.data) |v| {
        try std.testing.expectEqual(@as(f32, 5.0), v);
    }
}

test "broadcast column to matrix" {
    const allocator = std.testing.allocator;

    // Column [3,1] broadcast to [3,4]
    var col = try Tensor.init(allocator, &[_]usize{ 3, 1 });
    defer col.deinit();
    col.set(&[_]usize{ 0, 0 }, 1.0);
    col.set(&[_]usize{ 1, 0 }, 2.0);
    col.set(&[_]usize{ 2, 0 }, 3.0);

    var result = try broadcast(allocator, &col, &[_]usize{ 3, 4 });
    defer result.deinit();

    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 4 }, result.shape);

    // Each row should be replicated 4 times
    for (0..3) |row| {
        const expected: f32 = @floatFromInt(row + 1);
        for (0..4) |c| {
            const val = result.at(&[_]usize{ row, c });
            try std.testing.expectEqual(expected, val);
        }
    }
}

test "broadcast row to matrix" {
    const allocator = std.testing.allocator;

    // Row [1,4] broadcast to [3,4]
    var row = try Tensor.init(allocator, &[_]usize{ 1, 4 });
    defer row.deinit();
    for (0..4) |i| {
        row.set(&[_]usize{ 0, i }, @floatFromInt(i + 1));
    }

    var result = try broadcast(allocator, &row, &[_]usize{ 3, 4 });
    defer result.deinit();

    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 4 }, result.shape);

    // Each column should have the same value across rows
    for (0..3) |r| {
        for (0..4) |c| {
            const expected: f32 = @floatFromInt(c + 1);
            const val = result.at(&[_]usize{ r, c });
            try std.testing.expectEqual(expected, val);
        }
    }
}

test "broadcast 1D to 3D" {
    const allocator = std.testing.allocator;

    // [4] broadcast to [2,3,4]
    var vec = try Tensor.init(allocator, &[_]usize{4});
    defer vec.deinit();
    for (0..4) |i| {
        vec.data[i] = @floatFromInt(i);
    }

    var result = try broadcast(allocator, &vec, &[_]usize{ 2, 3, 4 });
    defer result.deinit();

    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 4 }, result.shape);

    // Check that pattern [0,1,2,3] repeats for each [batch, row]
    for (0..2) |b| {
        for (0..3) |r| {
            for (0..4) |c| {
                const expected: f32 = @floatFromInt(c);
                const val = result.at(&[_]usize{ b, r, c });
                try std.testing.expectEqual(expected, val);
            }
        }
    }
}

test "broadcastPair" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{ 3, 1 });
    defer a.deinit();
    a.set(&[_]usize{ 0, 0 }, 1.0);
    a.set(&[_]usize{ 1, 0 }, 2.0);
    a.set(&[_]usize{ 2, 0 }, 3.0);

    var b = try Tensor.init(allocator, &[_]usize{ 1, 4 });
    defer b.deinit();
    for (0..4) |i| {
        b.set(&[_]usize{ 0, i }, @floatFromInt(i + 10));
    }

    var pair = try broadcastPair(allocator, &a, &b);
    defer pair.a.deinit();
    defer pair.b.deinit();

    // Both should be [3,4]
    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 4 }, pair.a.shape);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 4 }, pair.b.shape);

    // pair.a: each row has same value (1, 2, 3)
    try std.testing.expectEqual(@as(f32, 1.0), pair.a.at(&[_]usize{ 0, 2 }));
    try std.testing.expectEqual(@as(f32, 3.0), pair.a.at(&[_]usize{ 2, 0 }));

    // pair.b: each column has same value (10, 11, 12, 13)
    try std.testing.expectEqual(@as(f32, 10.0), pair.b.at(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 13.0), pair.b.at(&[_]usize{ 2, 3 }));
}
