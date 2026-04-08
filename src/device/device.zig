const std = @import("std");
const Allocator = std.mem.Allocator;

/// Device types for tensor placement
pub const DeviceType = enum {
    cpu,
    gpu,
};

/// Device identifier
pub const Device = struct {
    device_type: DeviceType,
    index: usize,

    const Self = @This();

    pub const cpu = Self{ .device_type = .cpu, .index = 0 };

    pub fn gpu(index: usize) Self {
        return Self{ .device_type = .gpu, .index = index };
    }

    pub fn isCpu(self: Self) bool {
        return self.device_type == .cpu;
    }

    pub fn isGpu(self: Self) bool {
        return self.device_type == .gpu;
    }

    pub fn eql(self: Self, other: Self) bool {
        return self.device_type == other.device_type and self.index == other.index;
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        switch (self.device_type) {
            .cpu => try writer.print("cpu", .{}),
            .gpu => try writer.print("gpu:{}", .{self.index}),
        }
    }
};

/// GPU device information
pub const GpuInfo = struct {
    name: [256]u8,
    name_len: usize,
    total_memory: u64,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    multiprocessor_count: u32,
    max_threads_per_block: u32,
    max_shared_memory_per_block: u32,

    pub fn getName(self: *const GpuInfo) []const u8 {
        return self.name[0..self.name_len];
    }
};

/// GPU memory statistics
pub const MemoryStats = struct {
    total_bytes: u64,
    used_bytes: u64,
    free_bytes: u64,
    peak_used_bytes: u64,
    allocation_count: u64,
};

/// Backend interface for device operations
pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Allocate memory on this device
        alloc: *const fn (ctx: *anyopaque, size: usize) ?[*]u8,
        /// Free memory on this device
        free: *const fn (ctx: *anyopaque, ptr: [*]u8, size: usize) void,
        /// Copy data between devices
        copyTo: *const fn (ctx: *anyopaque, dst: Device, src_ptr: [*]const u8, size: usize) ?[*]u8,
        /// Synchronize device operations
        sync: *const fn (ctx: *anyopaque) void,
        /// Check if device is available
        isAvailable: *const fn (ctx: *anyopaque) bool,
        /// Get device info
        getInfo: *const fn (ctx: *anyopaque) ?GpuInfo,
        /// Get memory stats
        getMemoryStats: *const fn (ctx: *anyopaque) MemoryStats,
    };

    pub fn alloc(self: Backend, size: usize) ?[*]u8 {
        return self.vtable.alloc(self.ptr, size);
    }

    pub fn free(self: Backend, ptr: [*]u8, size: usize) void {
        self.vtable.free(self.ptr, ptr, size);
    }

    pub fn copyTo(self: Backend, dst: Device, src_ptr: [*]const u8, size: usize) ?[*]u8 {
        return self.vtable.copyTo(self.ptr, dst, src_ptr, size);
    }

    pub fn sync(self: Backend) void {
        self.vtable.sync(self.ptr);
    }

    pub fn isAvailable(self: Backend) bool {
        return self.vtable.isAvailable(self.ptr);
    }

    pub fn getInfo(self: Backend) ?GpuInfo {
        return self.vtable.getInfo(self.ptr);
    }

    pub fn getMemoryStats(self: Backend) MemoryStats {
        return self.vtable.getMemoryStats(self.ptr);
    }
};

const MAX_GPUS = 8;

/// Global device manager
pub const DeviceManager = struct {
    allocator: Allocator,
    cpu_backend: Backend,
    gpu_backends: [MAX_GPUS]?Backend,
    default_device: Device,
    gpu_count: usize,
    initialized: bool,

    const Self = @This();

    var instance: ?Self = null;

    pub fn init(allocator: Allocator) !*Self {
        if (instance) |*inst| {
            return inst;
        }

        const cpu = @import("cpu.zig");
        const gpu_mod = @import("gpu.zig");

        var gpu_backends = [_]?Backend{null} ** MAX_GPUS;

        // Detect GPUs
        const gpu_count = gpu_mod.detectGpus();
        for (0..@min(gpu_count, MAX_GPUS)) |i| {
            gpu_backends[i] = gpu_mod.initGpu(allocator, i);
        }

        instance = Self{
            .allocator = allocator,
            .cpu_backend = cpu.initCpu(allocator),
            .gpu_backends = gpu_backends,
            .default_device = Device.cpu,
            .gpu_count = gpu_count,
            .initialized = true,
        };

        return &instance.?;
    }

    pub fn deinit(self: *Self) void {
        const gpu_mod = @import("gpu.zig");
        for (self.gpu_backends) |maybe_backend| {
            if (maybe_backend) |backend| {
                gpu_mod.deinitGpu(backend);
            }
        }
        self.initialized = false;
        instance = null;
    }

    pub fn getBackend(self: *Self, device: Device) Backend {
        return switch (device.device_type) {
            .cpu => self.cpu_backend,
            .gpu => if (device.index < MAX_GPUS and self.gpu_backends[device.index] != null)
                self.gpu_backends[device.index].?
            else
                self.cpu_backend, // Fallback to CPU
        };
    }

    pub fn setDefaultDevice(self: *Self, device: Device) void {
        self.default_device = device;
    }

    pub fn getDefaultDevice(self: *Self) Device {
        return self.default_device;
    }

    pub fn gpuAvailable(self: *Self) bool {
        for (self.gpu_backends) |maybe_backend| {
            if (maybe_backend) |backend| {
                if (backend.isAvailable()) return true;
            }
        }
        return false;
    }

    pub fn getGpuCount(self: *Self) usize {
        return self.gpu_count;
    }
};

test "device creation" {
    const cpu_dev = Device.cpu;
    try std.testing.expect(cpu_dev.isCpu());
    try std.testing.expect(!cpu_dev.isGpu());

    const gpu0 = Device.gpu(0);
    try std.testing.expect(gpu0.isGpu());
    try std.testing.expect(!gpu0.isCpu());
    try std.testing.expectEqual(@as(usize, 0), gpu0.index);

    const gpu1 = Device.gpu(1);
    try std.testing.expectEqual(@as(usize, 1), gpu1.index);
}

test "device equality" {
    const cpu_dev = Device.cpu;
    const cpu2 = Device{ .device_type = .cpu, .index = 0 };
    try std.testing.expect(cpu_dev.eql(cpu2));

    const gpu0 = Device.gpu(0);
    const gpu1 = Device.gpu(1);
    try std.testing.expect(!gpu0.eql(gpu1));
    try std.testing.expect(!cpu_dev.eql(gpu0));
}
