const std = @import("std");
const Allocator = std.mem.Allocator;
const device_mod = @import("device.zig");
const memory = @import("memory.zig");
const Device = device_mod.Device;
const DeviceManager = device_mod.DeviceManager;
const MemoryLocation = memory.MemoryLocation;
const gpu = @import("gpu.zig");

/// GPU-aware tensor with automatic device management
pub const DeviceTensor = struct {
    shape: []usize,
    strides: []usize,
    location: MemoryLocation,
    allocator: Allocator,

    const Self = @This();

    /// Create tensor on specified device
    pub fn init(allocator: Allocator, shape: []const usize, dev: Device) !Self {
        const num_dims = shape.len;
        if (num_dims == 0) return error.InvalidShape;

        var total_size: usize = 1;
        for (shape) |dim| {
            if (dim == 0) return error.InvalidShape;
            total_size *= dim;
        }

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

        const byte_size = total_size * @sizeOf(f32);
        const loc = try memory.allocOnDevice(allocator, dev, byte_size);

        return Self{
            .shape = owned_shape,
            .strides = strides,
            .location = loc,
            .allocator = allocator,
        };
    }

    /// Create tensor on default device (CPU unless GPU is set)
    pub fn initDefault(allocator: Allocator, shape: []const usize) !Self {
        var dm = try DeviceManager.init(allocator);
        return init(allocator, shape, dm.getDefaultDevice());
    }

    /// Create tensor on CPU
    pub fn initCpu(allocator: Allocator, shape: []const usize) !Self {
        return init(allocator, shape, Device.cpu);
    }

    /// Create tensor on GPU with index
    pub fn initGpu(allocator: Allocator, shape: []const usize, gpu_index: usize) !Self {
        return init(allocator, shape, Device.gpu(gpu_index));
    }

    pub fn deinit(self: *Self) void {
        memory.freeOnDevice(self.allocator, self.location);
        self.allocator.free(self.strides);
        self.allocator.free(self.shape);
        self.* = undefined;
    }

    /// Get device where tensor is stored
    pub fn device(self: *const Self) Device {
        return self.location.device;
    }

    /// Check if tensor is on CPU
    pub fn isCpu(self: *const Self) bool {
        return self.location.device.isCpu();
    }

    /// Check if tensor is on GPU
    pub fn isGpu(self: *const Self) bool {
        return self.location.device.isGpu();
    }

    /// Get number of dimensions
    pub fn ndim(self: *const Self) usize {
        return self.shape.len;
    }

    /// Get total number of elements
    pub fn size(self: *const Self) usize {
        var s: usize = 1;
        for (self.shape) |dim| s *= dim;
        return s;
    }

    /// Get data as f32 pointer (only valid for CPU tensors)
    pub fn data(self: *Self) ![*]f32 {
        if (!self.isCpu()) {
            return error.NotOnCpu;
        }
        return @ptrCast(@alignCast(self.location.ptr));
    }

    /// Get data as const f32 pointer
    pub fn dataConst(self: *const Self) ![*]const f32 {
        if (!self.isCpu()) {
            return error.NotOnCpu;
        }
        return @ptrCast(@alignCast(self.location.ptr));
    }

    /// Move tensor to specified device
    pub fn to(self: *Self, target_device: Device) !void {
        if (self.location.device.eql(target_device)) return;

        const new_loc = try memory.transfer(self.allocator, self.location, target_device);
        memory.freeOnDevice(self.allocator, self.location);
        self.location = new_loc;
    }

    /// Move to CPU
    pub fn toCpu(self: *Self) !void {
        try self.to(Device.cpu);
    }

    /// Move to GPU
    pub fn toGpu(self: *Self, gpu_index: usize) !void {
        try self.to(Device.gpu(gpu_index));
    }

    /// Create a copy on the same device
    pub fn clone(self: *const Self) !Self {
        const new = try Self.init(self.allocator, self.shape, self.location.device);

        if (self.isCpu()) {
            const src: [*]const u8 = self.location.ptr;
            const dst: [*]u8 = new.location.ptr;
            @memcpy(dst[0..self.location.size], src[0..self.location.size]);
        } else {
            // GPU copy would use cuMemcpy or vkCmdCopyBuffer
            // For now, go through CPU
            const cpu_data = try memory.toCpuSlice(self.allocator, self.location);
            defer if (!self.isCpu()) self.allocator.free(cpu_data);

            const dst: [*]u8 = new.location.ptr;
            const src_bytes: [*]const u8 = @ptrCast(cpu_data.ptr);
            @memcpy(dst[0..self.location.size], src_bytes[0..self.location.size]);
        }

        return new;
    }

    /// Create a copy on a different device
    pub fn cloneTo(self: *const Self, target_device: Device) !Self {
        var new = try self.clone();
        try new.to(target_device);
        return new;
    }

    /// Fill with value (CPU only for now)
    pub fn fill(self: *Self, value: f32) !void {
        if (!self.isCpu()) {
            // Would use GPU fill kernel
            try self.toCpu();
        }

        const ptr = try self.data();
        const len = self.size();
        for (0..len) |i| {
            ptr[i] = value;
        }
    }

    /// Create tensor filled with zeros
    pub fn zeros(allocator: Allocator, shape: []const usize, dev: Device) !Self {
        var t = try Self.init(allocator, shape, dev);
        try t.fill(0.0);
        return t;
    }

    /// Create tensor filled with ones
    pub fn ones(allocator: Allocator, shape: []const usize, dev: Device) !Self {
        var t = try Self.init(allocator, shape, dev);
        try t.fill(1.0);
        return t;
    }

    /// Synchronize operations on this tensor's device
    pub fn sync(self: *const Self) void {
        memory.syncDevice(self.allocator, self.location.device);
    }
};

// ============================================================================
// GPU-accelerated operations with CPU fallback
// ============================================================================

/// Element-wise addition (GPU-accelerated when available)
pub fn add(a: *const DeviceTensor, b: *const DeviceTensor, out: *DeviceTensor) !void {
    if (!sameTensorShape(a, b) or !sameTensorShape(a, out)) {
        return error.ShapeMismatch;
    }

    // Try GPU kernel if all on same GPU
    if (a.isGpu() and b.isGpu() and out.isGpu() and
        a.location.device.eql(b.location.device) and
        a.location.device.eql(out.location.device))
    {
        const a_f32: [*]const f32 = @ptrCast(@alignCast(a.location.ptr));
        const b_f32: [*]const f32 = @ptrCast(@alignCast(b.location.ptr));
        const out_f32: [*]f32 = @ptrCast(@alignCast(out.location.ptr));

        if (gpu.gpuAdd(a_f32, b_f32, out_f32, a.size())) {
            return; // GPU succeeded
        }
    }

    // CPU fallback
    const a_data = try memory.toCpuSlice(a.allocator, a.location);
    const b_data = try memory.toCpuSlice(b.allocator, b.location);
    const out_data = try out.data();

    for (0..a.size()) |i| {
        out_data[i] = a_data[i] + b_data[i];
    }
}

/// Element-wise multiplication (GPU-accelerated when available)
pub fn mul(a: *const DeviceTensor, b: *const DeviceTensor, out: *DeviceTensor) !void {
    if (!sameTensorShape(a, b) or !sameTensorShape(a, out)) {
        return error.ShapeMismatch;
    }

    // Try GPU kernel
    if (a.isGpu() and b.isGpu() and out.isGpu() and
        a.location.device.eql(b.location.device) and
        a.location.device.eql(out.location.device))
    {
        const a_f32: [*]const f32 = @ptrCast(@alignCast(a.location.ptr));
        const b_f32: [*]const f32 = @ptrCast(@alignCast(b.location.ptr));
        const out_f32: [*]f32 = @ptrCast(@alignCast(out.location.ptr));

        if (gpu.gpuMul(a_f32, b_f32, out_f32, a.size())) {
            return;
        }
    }

    // CPU fallback
    const a_data = try memory.toCpuSlice(a.allocator, a.location);
    const b_data = try memory.toCpuSlice(b.allocator, b.location);
    const out_data = try out.data();

    for (0..a.size()) |i| {
        out_data[i] = a_data[i] * b_data[i];
    }
}

/// Matrix multiplication (GPU-accelerated when available)
pub fn matmul(a: *const DeviceTensor, b: *const DeviceTensor, out: *DeviceTensor) !void {
    if (a.ndim() != 2 or b.ndim() != 2 or out.ndim() != 2) {
        return error.InvalidDimensions;
    }

    const m = a.shape[0];
    const k = a.shape[1];
    const n = b.shape[1];

    if (b.shape[0] != k or out.shape[0] != m or out.shape[1] != n) {
        return error.ShapeMismatch;
    }

    // Try GPU kernel
    if (a.isGpu() and b.isGpu() and out.isGpu() and
        a.location.device.eql(b.location.device) and
        a.location.device.eql(out.location.device))
    {
        const a_f32: [*]const f32 = @ptrCast(@alignCast(a.location.ptr));
        const b_f32: [*]const f32 = @ptrCast(@alignCast(b.location.ptr));
        const out_f32: [*]f32 = @ptrCast(@alignCast(out.location.ptr));

        if (gpu.gpuMatmul(a_f32, b_f32, out_f32, m, k, n)) {
            return;
        }
    }

    // CPU fallback (naive triple loop)
    const a_data = try memory.toCpuSlice(a.allocator, a.location);
    const b_data = try memory.toCpuSlice(b.allocator, b.location);
    const out_data = try out.data();

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |p| {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }
}

fn sameTensorShape(a: *const DeviceTensor, b: *const DeviceTensor) bool {
    if (a.shape.len != b.shape.len) return false;
    for (a.shape, b.shape) |x, y| {
        if (x != y) return false;
    }
    return true;
}

test "device tensor creation cpu" {
    const allocator = std.testing.allocator;

    var t = try DeviceTensor.initCpu(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();

    try std.testing.expect(t.isCpu());
    try std.testing.expectEqual(@as(usize, 2), t.ndim());
    try std.testing.expectEqual(@as(usize, 6), t.size());
}

test "device tensor zeros" {
    const allocator = std.testing.allocator;

    var t = try DeviceTensor.zeros(allocator, &[_]usize{4}, Device.cpu);
    defer t.deinit();

    const data = try t.data();
    for (0..4) |i| {
        try std.testing.expectEqual(@as(f32, 0.0), data[i]);
    }
}

test "device tensor fill" {
    const allocator = std.testing.allocator;

    var t = try DeviceTensor.initCpu(allocator, &[_]usize{4});
    defer t.deinit();

    try t.fill(3.14);
    const data = try t.data();
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 3.14), data[i], 0.001);
    }
}

test "device tensor clone" {
    const allocator = std.testing.allocator;

    var t = try DeviceTensor.zeros(allocator, &[_]usize{ 2, 2 }, Device.cpu);
    defer t.deinit();

    const data = try t.data();
    data[0] = 42.0;

    var c = try t.clone();
    defer c.deinit();

    const c_data = try c.data();
    try std.testing.expectEqual(@as(f32, 42.0), c_data[0]);

    // Modify original, clone should be independent
    data[0] = 0.0;
    try std.testing.expectEqual(@as(f32, 42.0), c_data[0]);
}

test "device tensor add" {
    const allocator = std.testing.allocator;

    var a = try DeviceTensor.initCpu(allocator, &[_]usize{4});
    defer a.deinit();
    var b = try DeviceTensor.initCpu(allocator, &[_]usize{4});
    defer b.deinit();
    var out = try DeviceTensor.initCpu(allocator, &[_]usize{4});
    defer out.deinit();

    const a_data = try a.data();
    const b_data = try b.data();
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;
    b_data[0] = 10.0;
    b_data[1] = 20.0;
    b_data[2] = 30.0;
    b_data[3] = 40.0;

    try add(&a, &b, &out);

    const out_data = try out.data();
    try std.testing.expectEqual(@as(f32, 11.0), out_data[0]);
    try std.testing.expectEqual(@as(f32, 22.0), out_data[1]);
    try std.testing.expectEqual(@as(f32, 33.0), out_data[2]);
    try std.testing.expectEqual(@as(f32, 44.0), out_data[3]);
}

test "device tensor matmul" {
    const allocator = std.testing.allocator;

    // [2x3] @ [3x2] = [2x2]
    var a = try DeviceTensor.initCpu(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();
    var b = try DeviceTensor.initCpu(allocator, &[_]usize{ 3, 2 });
    defer b.deinit();
    var out = try DeviceTensor.initCpu(allocator, &[_]usize{ 2, 2 });
    defer out.deinit();

    // a = [[1, 2, 3], [4, 5, 6]]
    const a_data = try a.data();
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;
    a_data[4] = 5.0;
    a_data[5] = 6.0;

    // b = [[1, 2], [3, 4], [5, 6]]
    const b_data = try b.data();
    b_data[0] = 1.0;
    b_data[1] = 2.0;
    b_data[2] = 3.0;
    b_data[3] = 4.0;
    b_data[4] = 5.0;
    b_data[5] = 6.0;

    try matmul(&a, &b, &out);

    // out = [[22, 28], [49, 64]]
    const out_data = try out.data();
    try std.testing.expectEqual(@as(f32, 22.0), out_data[0]);
    try std.testing.expectEqual(@as(f32, 28.0), out_data[1]);
    try std.testing.expectEqual(@as(f32, 49.0), out_data[2]);
    try std.testing.expectEqual(@as(f32, 64.0), out_data[3]);
}
