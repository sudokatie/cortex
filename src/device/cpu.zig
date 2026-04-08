const std = @import("std");
const Allocator = std.mem.Allocator;
const device = @import("device.zig");
const Backend = device.Backend;
const Device = device.Device;
const GpuInfo = device.GpuInfo;
const MemoryStats = device.MemoryStats;

/// CPU backend state
pub const CpuBackend = struct {
    allocator: Allocator,
    total_allocated: u64,
    peak_allocated: u64,
    allocation_count: u64,

    const Self = @This();

    fn alloc(ctx: *anyopaque, size: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const slice = self.allocator.alloc(u8, size) catch return null;
        self.total_allocated += size;
        if (self.total_allocated > self.peak_allocated) {
            self.peak_allocated = self.total_allocated;
        }
        self.allocation_count += 1;
        return slice.ptr;
    }

    fn free(ctx: *anyopaque, ptr: [*]u8, size: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const slice = ptr[0..size];
        self.allocator.free(slice);
        self.total_allocated -= size;
    }

    fn copyTo(ctx: *anyopaque, dst: Device, src_ptr: [*]const u8, size: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = dst;

        // For CPU->CPU, just allocate and copy
        const dest_slice = self.allocator.alloc(u8, size) catch return null;
        @memcpy(dest_slice, src_ptr[0..size]);
        self.total_allocated += size;
        if (self.total_allocated > self.peak_allocated) {
            self.peak_allocated = self.total_allocated;
        }
        self.allocation_count += 1;
        return dest_slice.ptr;
    }

    fn sync(ctx: *anyopaque) void {
        _ = ctx;
        // CPU operations are synchronous, nothing to do
    }

    fn isAvailable(ctx: *anyopaque) bool {
        _ = ctx;
        return true; // CPU is always available
    }

    fn getInfo(ctx: *anyopaque) ?GpuInfo {
        _ = ctx;
        return null; // CPU doesn't have GPU info
    }

    fn getMemoryStats(ctx: *anyopaque) MemoryStats {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return MemoryStats{
            .total_bytes = std.math.maxInt(u64), // CPU memory is "unlimited"
            .used_bytes = self.total_allocated,
            .free_bytes = std.math.maxInt(u64) - self.total_allocated,
            .peak_used_bytes = self.peak_allocated,
            .allocation_count = self.allocation_count,
        };
    }
};

var cpu_backend_instance: ?CpuBackend = null;

const vtable = Backend.VTable{
    .alloc = CpuBackend.alloc,
    .free = CpuBackend.free,
    .copyTo = CpuBackend.copyTo,
    .sync = CpuBackend.sync,
    .isAvailable = CpuBackend.isAvailable,
    .getInfo = CpuBackend.getInfo,
    .getMemoryStats = CpuBackend.getMemoryStats,
};

/// Initialize CPU backend
pub fn initCpu(allocator: Allocator) Backend {
    if (cpu_backend_instance == null) {
        cpu_backend_instance = CpuBackend{
            .allocator = allocator,
            .total_allocated = 0,
            .peak_allocated = 0,
            .allocation_count = 0,
        };
    }
    return Backend{
        .ptr = &cpu_backend_instance.?,
        .vtable = &vtable,
    };
}

test "cpu backend allocation" {
    const allocator = std.testing.allocator;
    const backend = initCpu(allocator);

    const stats_before = backend.getMemoryStats();
    const used_before = stats_before.used_bytes;
    const count_before = stats_before.allocation_count;

    const ptr = backend.alloc(100);
    try std.testing.expect(ptr != null);

    const stats_after = backend.getMemoryStats();
    try std.testing.expectEqual(used_before + 100, stats_after.used_bytes);
    try std.testing.expectEqual(count_before + 1, stats_after.allocation_count);

    backend.free(ptr.?, 100);
}

test "cpu backend always available" {
    const allocator = std.testing.allocator;
    const backend = initCpu(allocator);
    try std.testing.expect(backend.isAvailable());
}

test "cpu backend no gpu info" {
    const allocator = std.testing.allocator;
    const backend = initCpu(allocator);
    try std.testing.expect(backend.getInfo() == null);
}
