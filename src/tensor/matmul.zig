const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

/// Matrix multiplication: C = A @ B
/// A is [M, K], B is [K, N], result is [M, N]
pub fn matmul(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (a.ndim() != 2 or b.ndim() != 2) {
        return error.InvalidDimensions;
    }

    const m = a.shape[0];
    const k1 = a.shape[1];
    const k2 = b.shape[0];
    const n = b.shape[1];

    if (k1 != k2) {
        return error.DimensionMismatch;
    }

    const k = k1;
    const result = try Tensor.zeros(allocator, &[_]usize{ m, n });

    // C[i,j] = sum_k A[i,k] * B[k,j]
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| {
                const a_val = a.at(&[_]usize{ i, kk });
                const b_val = b.at(&[_]usize{ kk, j });
                sum += a_val * b_val;
            }
            result.data[i * n + j] = sum;
        }
    }

    return result;
}

/// Matrix multiplication with A transposed: C = A^T @ B
/// A is [K, M], B is [K, N], result is [M, N]
pub fn matmulTransposeA(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (a.ndim() != 2 or b.ndim() != 2) {
        return error.InvalidDimensions;
    }

    const k1 = a.shape[0];
    const m = a.shape[1];
    const k2 = b.shape[0];
    const n = b.shape[1];

    if (k1 != k2) {
        return error.DimensionMismatch;
    }

    const k = k1;
    const result = try Tensor.zeros(allocator, &[_]usize{ m, n });

    // C[i,j] = sum_k A[k,i] * B[k,j]
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| {
                const a_val = a.at(&[_]usize{ kk, i }); // transposed access
                const b_val = b.at(&[_]usize{ kk, j });
                sum += a_val * b_val;
            }
            result.data[i * n + j] = sum;
        }
    }

    return result;
}

/// Matrix multiplication with B transposed: C = A @ B^T
/// A is [M, K], B is [N, K], result is [M, N]
pub fn matmulTransposeB(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (a.ndim() != 2 or b.ndim() != 2) {
        return error.InvalidDimensions;
    }

    const m = a.shape[0];
    const k1 = a.shape[1];
    const n = b.shape[0];
    const k2 = b.shape[1];

    if (k1 != k2) {
        return error.DimensionMismatch;
    }

    const k = k1;
    const result = try Tensor.zeros(allocator, &[_]usize{ m, n });

    // C[i,j] = sum_k A[i,k] * B[j,k]
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |kk| {
                const a_val = a.at(&[_]usize{ i, kk });
                const b_val = b.at(&[_]usize{ j, kk }); // transposed access
                sum += a_val * b_val;
            }
            result.data[i * n + j] = sum;
        }
    }

    return result;
}

// ============== Tests ==============

test "matmul basic" {
    const allocator = std.testing.allocator;

    // A = [[1, 2], [3, 4]]  (2x2)
    var a = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;

    // B = [[5, 6], [7, 8]]  (2x2)
    var b = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();
    b.data[0] = 5.0;
    b.data[1] = 6.0;
    b.data[2] = 7.0;
    b.data[3] = 8.0;

    // C = A @ B = [[19, 22], [43, 50]]
    var c = try matmul(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 19.0), c.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), c.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 43.0), c.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), c.data[3], 0.001);
}

test "matmul non-square" {
    const allocator = std.testing.allocator;

    // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
    var a = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    a.data[4] = 5.0;
    a.data[5] = 6.0;

    // B = [[1, 2], [3, 4], [5, 6]]  (3x2)
    var b = try Tensor.init(allocator, &[_]usize{ 3, 2 });
    defer b.deinit();
    b.data[0] = 1.0;
    b.data[1] = 2.0;
    b.data[2] = 3.0;
    b.data[3] = 4.0;
    b.data[4] = 5.0;
    b.data[5] = 6.0;

    // C = A @ B = [[22, 28], [49, 64]]  (2x2)
    var c = try matmul(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectEqual(@as(usize, 2), c.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), c.shape[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), c.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 28.0), c.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 49.0), c.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), c.data[3], 0.001);
}

test "matmul transpose A" {
    const allocator = std.testing.allocator;

    // A = [[1, 3], [2, 4]]  (2x2), A^T = [[1, 2], [3, 4]]
    var a = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 3.0;
    a.data[2] = 2.0;
    a.data[3] = 4.0;

    // B = [[5, 6], [7, 8]]  (2x2)
    var b = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();
    b.data[0] = 5.0;
    b.data[1] = 6.0;
    b.data[2] = 7.0;
    b.data[3] = 8.0;

    // C = A^T @ B = [[19, 22], [43, 50]]
    var c = try matmulTransposeA(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 19.0), c.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), c.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 43.0), c.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), c.data[3], 0.001);
}

test "matmul dimension mismatch" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();

    var b = try Tensor.init(allocator, &[_]usize{ 2, 2 }); // k mismatch: 3 != 2
    defer b.deinit();

    const result = matmul(allocator, &a, &b);
    try std.testing.expectError(error.DimensionMismatch, result);
}
