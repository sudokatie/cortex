const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

/// Sum all elements (when axis is null) or along an axis.
pub fn sum(allocator: Allocator, t: *const Tensor, axis: ?usize) !Tensor {
    if (axis == null) {
        // Sum all elements
        var total: f32 = 0.0;
        for (t.data) |v| {
            total += v;
        }
        const result = try Tensor.init(allocator, &[_]usize{1});
        result.data[0] = total;
        return result;
    }

    const ax = axis.?;
    if (ax >= t.ndim()) {
        return error.InvalidAxis;
    }

    // Build result shape (remove the axis dimension)
    var result_shape = try allocator.alloc(usize, t.ndim() - 1);
    defer allocator.free(result_shape);

    var j: usize = 0;
    for (0..t.ndim()) |i| {
        if (i != ax) {
            result_shape[j] = t.shape[i];
            j += 1;
        }
    }

    const result = try Tensor.zeros(allocator, result_shape);

    // For 2D case (most common)
    if (t.ndim() == 2) {
        const rows = t.shape[0];
        const cols = t.shape[1];

        if (ax == 0) {
            // Sum along rows -> result shape [cols]
            for (0..cols) |c| {
                var s: f32 = 0.0;
                for (0..rows) |r| {
                    s += t.at(&[_]usize{ r, c });
                }
                result.data[c] = s;
            }
        } else {
            // Sum along cols -> result shape [rows]
            for (0..rows) |r| {
                var s: f32 = 0.0;
                for (0..cols) |c| {
                    s += t.at(&[_]usize{ r, c });
                }
                result.data[r] = s;
            }
        }
    }

    return result;
}

/// Mean of all elements (when axis is null) or along an axis.
pub fn mean(allocator: Allocator, t: *const Tensor, axis: ?usize) !Tensor {
    const result = try sum(allocator, t, axis);

    const count: f32 = if (axis == null)
        @floatFromInt(t.size())
    else
        @floatFromInt(t.shape[axis.?]);

    for (result.data) |*v| {
        v.* /= count;
    }

    return result;
}

/// Max of all elements (when axis is null) or along an axis.
pub fn max(allocator: Allocator, t: *const Tensor, axis: ?usize) !Tensor {
    if (axis == null) {
        // Max of all elements
        var m: f32 = t.data[0];
        for (t.data[1..]) |v| {
            if (v > m) m = v;
        }
        const result = try Tensor.init(allocator, &[_]usize{1});
        result.data[0] = m;
        return result;
    }

    const ax = axis.?;
    if (ax >= t.ndim()) {
        return error.InvalidAxis;
    }

    // Build result shape
    var result_shape = try allocator.alloc(usize, t.ndim() - 1);
    defer allocator.free(result_shape);

    var j: usize = 0;
    for (0..t.ndim()) |i| {
        if (i != ax) {
            result_shape[j] = t.shape[i];
            j += 1;
        }
    }

    const result = try Tensor.init(allocator, result_shape);

    // For 2D case
    if (t.ndim() == 2) {
        const rows = t.shape[0];
        const cols = t.shape[1];

        if (ax == 0) {
            // Max along rows -> result shape [cols]
            for (0..cols) |c| {
                var m: f32 = t.at(&[_]usize{ 0, c });
                for (1..rows) |r| {
                    const v = t.at(&[_]usize{ r, c });
                    if (v > m) m = v;
                }
                result.data[c] = m;
            }
        } else {
            // Max along cols -> result shape [rows]
            for (0..rows) |r| {
                var m: f32 = t.at(&[_]usize{ r, 0 });
                for (1..cols) |c| {
                    const v = t.at(&[_]usize{ r, c });
                    if (v > m) m = v;
                }
                result.data[r] = m;
            }
        }
    }

    return result;
}

/// Argmax of all elements (when axis is null) or along an axis.
/// Returns indices as f32 for consistency.
pub fn argmax(allocator: Allocator, t: *const Tensor, axis: ?usize) !Tensor {
    if (axis == null) {
        // Argmax of all elements
        var max_idx: usize = 0;
        var max_val: f32 = t.data[0];
        for (t.data[1..], 1..) |v, i| {
            if (v > max_val) {
                max_val = v;
                max_idx = i;
            }
        }
        const result = try Tensor.init(allocator, &[_]usize{1});
        result.data[0] = @floatFromInt(max_idx);
        return result;
    }

    const ax = axis.?;
    if (ax >= t.ndim()) {
        return error.InvalidAxis;
    }

    // Build result shape
    var result_shape = try allocator.alloc(usize, t.ndim() - 1);
    defer allocator.free(result_shape);

    var j: usize = 0;
    for (0..t.ndim()) |i| {
        if (i != ax) {
            result_shape[j] = t.shape[i];
            j += 1;
        }
    }

    const result = try Tensor.init(allocator, result_shape);

    // For 2D case
    if (t.ndim() == 2) {
        const rows = t.shape[0];
        const cols = t.shape[1];

        if (ax == 0) {
            // Argmax along rows
            for (0..cols) |c| {
                var max_idx: usize = 0;
                var max_val: f32 = t.at(&[_]usize{ 0, c });
                for (1..rows) |r| {
                    const v = t.at(&[_]usize{ r, c });
                    if (v > max_val) {
                        max_val = v;
                        max_idx = r;
                    }
                }
                result.data[c] = @floatFromInt(max_idx);
            }
        } else {
            // Argmax along cols
            for (0..rows) |r| {
                var max_idx: usize = 0;
                var max_val: f32 = t.at(&[_]usize{ r, 0 });
                for (1..cols) |c| {
                    const v = t.at(&[_]usize{ r, c });
                    if (v > max_val) {
                        max_val = v;
                        max_idx = c;
                    }
                }
                result.data[r] = @floatFromInt(max_idx);
            }
        }
    }

    return result;
}

// ============== Tests ==============

test "sum all" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;
    t.data[3] = 4.0;
    t.data[4] = 5.0;
    t.data[5] = 6.0;

    var s = try sum(allocator, &t, null);
    defer s.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 21.0), s.data[0], 0.001);
}

test "sum axis 0" {
    const allocator = std.testing.allocator;

    // [[1, 2, 3], [4, 5, 6]]
    var t = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;
    t.data[3] = 4.0;
    t.data[4] = 5.0;
    t.data[5] = 6.0;

    // Sum along axis 0 -> [5, 7, 9]
    var s = try sum(allocator, &t, 0);
    defer s.deinit();

    try std.testing.expectEqual(@as(usize, 1), s.ndim());
    try std.testing.expectEqual(@as(usize, 3), s.shape[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), s.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), s.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), s.data[2], 0.001);
}

test "sum axis 1" {
    const allocator = std.testing.allocator;

    // [[1, 2, 3], [4, 5, 6]]
    var t = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;
    t.data[3] = 4.0;
    t.data[4] = 5.0;
    t.data[5] = 6.0;

    // Sum along axis 1 -> [6, 15]
    var s = try sum(allocator, &t, 1);
    defer s.deinit();

    try std.testing.expectEqual(@as(usize, 1), s.ndim());
    try std.testing.expectEqual(@as(usize, 2), s.shape[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), s.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), s.data[1], 0.001);
}

test "mean" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &[_]usize{4});
    defer t.deinit();
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;
    t.data[3] = 4.0;

    var m = try mean(allocator, &t, null);
    defer m.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), m.data[0], 0.001);
}

test "max" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &[_]usize{5});
    defer t.deinit();
    t.data[0] = 3.0;
    t.data[1] = 1.0;
    t.data[2] = 4.0;
    t.data[3] = 1.0;
    t.data[4] = 5.0;

    var m = try max(allocator, &t, null);
    defer m.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), m.data[0], 0.001);
}

test "argmax" {
    const allocator = std.testing.allocator;

    var t = try Tensor.init(allocator, &[_]usize{5});
    defer t.deinit();
    t.data[0] = 3.0;
    t.data[1] = 1.0;
    t.data[2] = 4.0;
    t.data[3] = 1.0;
    t.data[4] = 5.0;

    var am = try argmax(allocator, &t, null);
    defer am.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), am.data[0], 0.001);
}
