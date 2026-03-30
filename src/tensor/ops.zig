const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

/// Add two tensors element-wise. Must have same shape.
pub fn add(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (!a.sameShape(b)) {
        return error.ShapeMismatch;
    }

    const result = try Tensor.init(allocator, a.shape);
    for (result.data, a.data, b.data) |*r, av, bv| {
        r.* = av + bv;
    }
    return result;
}

/// Subtract two tensors element-wise. Must have same shape.
pub fn sub(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (!a.sameShape(b)) {
        return error.ShapeMismatch;
    }

    const result = try Tensor.init(allocator, a.shape);
    for (result.data, a.data, b.data) |*r, av, bv| {
        r.* = av - bv;
    }
    return result;
}

/// Multiply two tensors element-wise. Must have same shape.
pub fn mul(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (!a.sameShape(b)) {
        return error.ShapeMismatch;
    }

    const result = try Tensor.init(allocator, a.shape);
    for (result.data, a.data, b.data) |*r, av, bv| {
        r.* = av * bv;
    }
    return result;
}

/// Divide two tensors element-wise. Must have same shape.
pub fn div(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
    if (!a.sameShape(b)) {
        return error.ShapeMismatch;
    }

    const result = try Tensor.init(allocator, a.shape);
    for (result.data, a.data, b.data) |*r, av, bv| {
        r.* = av / bv;
    }
    return result;
}

/// Negate tensor element-wise.
pub fn neg(allocator: Allocator, t: *const Tensor) !Tensor {
    const result = try Tensor.init(allocator, t.shape);
    for (result.data, t.data) |*r, v| {
        r.* = -v;
    }
    return result;
}

/// Scale tensor by a scalar.
pub fn scale(allocator: Allocator, t: *const Tensor, s: f32) !Tensor {
    const result = try Tensor.init(allocator, t.shape);
    for (result.data, t.data) |*r, v| {
        r.* = v * s;
    }
    return result;
}

/// Element-wise exponential.
pub fn exp(allocator: Allocator, t: *const Tensor) !Tensor {
    const result = try Tensor.init(allocator, t.shape);
    for (result.data, t.data) |*r, v| {
        r.* = @exp(v);
    }
    return result;
}

/// Element-wise natural logarithm.
pub fn log(allocator: Allocator, t: *const Tensor) !Tensor {
    const result = try Tensor.init(allocator, t.shape);
    for (result.data, t.data) |*r, v| {
        r.* = @log(v);
    }
    return result;
}

/// Element-wise square root.
pub fn sqrt(allocator: Allocator, t: *const Tensor) !Tensor {
    const result = try Tensor.init(allocator, t.shape);
    for (result.data, t.data) |*r, v| {
        r.* = @sqrt(v);
    }
    return result;
}

/// Element-wise absolute value.
pub fn abs(allocator: Allocator, t: *const Tensor) !Tensor {
    const result = try Tensor.init(allocator, t.shape);
    for (result.data, t.data) |*r, v| {
        r.* = @abs(v);
    }
    return result;
}

/// Clamp values to [min, max] range.
pub fn clamp(allocator: Allocator, t: *const Tensor, min_val: f32, max_val: f32) !Tensor {
    const result = try Tensor.init(allocator, t.shape);
    for (result.data, t.data) |*r, v| {
        r.* = @max(min_val, @min(max_val, v));
    }
    return result;
}

/// Element-wise power.
pub fn pow(allocator: Allocator, t: *const Tensor, p: f32) !Tensor {
    const result = try Tensor.init(allocator, t.shape);
    for (result.data, t.data) |*r, v| {
        r.* = std.math.pow(f32, v, p);
    }
    return result;
}

// ============== Tests ==============

test "add" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{4});
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;

    var b = try Tensor.init(allocator, &[_]usize{4});
    defer b.deinit();
    b.data[0] = 5.0;
    b.data[1] = 6.0;
    b.data[2] = 7.0;
    b.data[3] = 8.0;

    var c = try add(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 6.0), c.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), c.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), c.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), c.data[3], 0.001);
}

test "sub" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{3});
    defer a.deinit();
    a.data[0] = 5.0;
    a.data[1] = 3.0;
    a.data[2] = 1.0;

    var b = try Tensor.init(allocator, &[_]usize{3});
    defer b.deinit();
    b.data[0] = 1.0;
    b.data[1] = 2.0;
    b.data[2] = 3.0;

    var c = try sub(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), c.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), c.data[2], 0.001);
}

test "mul" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{3});
    defer a.deinit();
    a.data[0] = 2.0;
    a.data[1] = 3.0;
    a.data[2] = 4.0;

    var b = try Tensor.init(allocator, &[_]usize{3});
    defer b.deinit();
    b.data[0] = 5.0;
    b.data[1] = 6.0;
    b.data[2] = 7.0;

    var c = try mul(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 10.0), c.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), c.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 28.0), c.data[2], 0.001);
}

test "div" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{2});
    defer a.deinit();
    a.data[0] = 10.0;
    a.data[1] = 20.0;

    var b = try Tensor.init(allocator, &[_]usize{2});
    defer b.deinit();
    b.data[0] = 2.0;
    b.data[1] = 4.0;

    var c = try div(allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), c.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), c.data[1], 0.001);
}

test "neg" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{3});
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = -2.0;
    a.data[2] = 3.0;

    var b = try neg(allocator, &a);
    defer b.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), b.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), b.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), b.data[2], 0.001);
}

test "scale" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{3});
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;

    var b = try scale(allocator, &a, 2.5);
    defer b.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), b.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), b.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), b.data[2], 0.001);
}

test "exp" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{2});
    defer a.deinit();
    a.data[0] = 0.0;
    a.data[1] = 1.0;

    var b = try exp(allocator, &a);
    defer b.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.718), b.data[1], 0.01);
}

test "log" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{2});
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = std.math.e;

    var b = try log(allocator, &a);
    defer b.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), b.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b.data[1], 0.001);
}

test "sqrt" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{3});
    defer a.deinit();
    a.data[0] = 4.0;
    a.data[1] = 9.0;
    a.data[2] = 16.0;

    var b = try sqrt(allocator, &a);
    defer b.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), b.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), b.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), b.data[2], 0.001);
}

test "abs" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{3});
    defer a.deinit();
    a.data[0] = -1.0;
    a.data[1] = 2.0;
    a.data[2] = -3.0;

    var b = try abs(allocator, &a);
    defer b.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), b.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), b.data[2], 0.001);
}

test "clamp" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{5});
    defer a.deinit();
    a.data[0] = -5.0;
    a.data[1] = 0.0;
    a.data[2] = 0.5;
    a.data[3] = 1.0;
    a.data[4] = 5.0;

    var b = try clamp(allocator, &a, 0.0, 1.0);
    defer b.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), b.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), b.data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), b.data[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b.data[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), b.data[4], 0.001);
}

test "shape mismatch" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, &[_]usize{3});
    defer a.deinit();

    var b = try Tensor.init(allocator, &[_]usize{4});
    defer b.deinit();

    const result = add(allocator, &a, &b);
    try std.testing.expectError(error.ShapeMismatch, result);
}
