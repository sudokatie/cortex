const std = @import("std");
const Allocator = std.mem.Allocator;
const device_mod = @import("device.zig");
const Device = device_mod.Device;
const DeviceManager = device_mod.DeviceManager;
const Backend = device_mod.Backend;

/// Memory location tracking for tensor data
pub const MemoryLocation = struct {
    device: Device,
    ptr: [*]u8,
    size: usize,
    is_view: bool, // If true, don't free on deinit

    const Self = @This();

    pub fn init(dev: Device, ptr: [*]u8, size: usize) Self {
        return Self{
            .device = dev,
            .ptr = ptr,
            .size = size,
            .is_view = false,
        };
    }

    pub fn initView(dev: Device, ptr: [*]u8, size: usize) Self {
        return Self{
            .device = dev,
            .ptr = ptr,
            .size = size,
            .is_view = true,
        };
    }
};

/// Transfer data between devices
pub fn transfer(
    allocator: Allocator,
    src: MemoryLocation,
    dst_device: Device,
) !MemoryLocation {
    if (src.device.eql(dst_device)) {
        // Same device, just return a view or copy
        return src;
    }

    var dm = try DeviceManager.init(allocator);
    const src_backend = dm.getBackend(src.device);
    const dst_backend = dm.getBackend(dst_device);

    // Transfer via source backend's copyTo
    if (src_backend.copyTo(dst_device, src.ptr, src.size)) |dst_ptr| {
        return MemoryLocation.init(dst_device, dst_ptr, src.size);
    }

    // Try via destination backend
    if (dst_backend.copyTo(dst_device, src.ptr, src.size)) |dst_ptr| {
        return MemoryLocation.init(dst_device, dst_ptr, src.size);
    }

    // Both failed, use CPU as intermediate
    if (!src.device.isCpu() and !dst_device.isCpu()) {
        // GPU -> CPU -> GPU
        const cpu_backend = dm.getBackend(Device.cpu);
        if (cpu_backend.copyTo(Device.cpu, src.ptr, src.size)) |cpu_ptr| {
            defer cpu_backend.free(cpu_ptr, src.size);
            if (dst_backend.copyTo(dst_device, cpu_ptr, src.size)) |final_ptr| {
                return MemoryLocation.init(dst_device, final_ptr, src.size);
            }
        }
    }

    return error.TransferFailed;
}

/// Copy data to CPU synchronously (for debugging/printing)
pub fn toCpuSlice(allocator: Allocator, loc: MemoryLocation) ![]f32 {
    if (loc.device.isCpu()) {
        // Already on CPU, return view as f32 slice
        const f32_ptr: [*]f32 = @ptrCast(@alignCast(loc.ptr));
        return f32_ptr[0 .. loc.size / @sizeOf(f32)];
    }

    // Transfer to CPU
    const cpu_loc = try transfer(allocator, loc, Device.cpu);
    const f32_ptr: [*]f32 = @ptrCast(@alignCast(cpu_loc.ptr));
    return f32_ptr[0 .. loc.size / @sizeOf(f32)];
}

/// Allocate memory on specified device
pub fn allocOnDevice(allocator: Allocator, dev: Device, size: usize) !MemoryLocation {
    var dm = try DeviceManager.init(allocator);
    const backend = dm.getBackend(dev);

    if (backend.alloc(size)) |ptr| {
        return MemoryLocation.init(dev, ptr, size);
    }

    // Try CPU fallback
    if (!dev.isCpu()) {
        const cpu_backend = dm.getBackend(Device.cpu);
        if (cpu_backend.alloc(size)) |ptr| {
            return MemoryLocation.init(Device.cpu, ptr, size);
        }
    }

    return error.OutOfMemory;
}

/// Free memory on its device
pub fn freeOnDevice(allocator: Allocator, loc: MemoryLocation) void {
    if (loc.is_view) return; // Don't free views

    var dm = DeviceManager.init(allocator) catch return;
    const backend = dm.getBackend(loc.device);
    backend.free(loc.ptr, loc.size);
}

/// Synchronize all pending operations on a device
pub fn syncDevice(allocator: Allocator, dev: Device) void {
    var dm = DeviceManager.init(allocator) catch return;
    const backend = dm.getBackend(dev);
    backend.sync();
}

test "memory location creation" {
    var buf: [100]u8 = undefined;
    const loc = MemoryLocation.init(Device.cpu, &buf, 100);

    try std.testing.expect(loc.device.isCpu());
    try std.testing.expectEqual(@as(usize, 100), loc.size);
    try std.testing.expect(!loc.is_view);
}

test "memory view creation" {
    var buf: [100]u8 = undefined;
    const loc = MemoryLocation.initView(Device.cpu, &buf, 100);

    try std.testing.expect(loc.is_view);
}
