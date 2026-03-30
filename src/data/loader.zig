const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Dataset for storing input-output pairs
pub const Dataset = struct {
    inputs: Tensor,
    targets: Tensor,
    num_samples: usize,

    const Self = @This();

    pub fn init(inputs: Tensor, targets: Tensor) !Self {
        if (inputs.shape[0] != targets.shape[0]) {
            return error.SampleCountMismatch;
        }
        return Self{
            .inputs = inputs,
            .targets = targets,
            .num_samples = inputs.shape[0],
        };
    }

    pub fn deinit(self: *Self) void {
        self.inputs.deinit();
        self.targets.deinit();
        self.* = undefined;
    }
};

/// Data loader for batching and shuffling
pub const DataLoader = struct {
    allocator: Allocator,
    dataset: *const Dataset,
    batch_size: usize,
    shuffle: bool,
    indices: []usize,
    current_idx: usize,
    rng: std.Random.DefaultPrng,

    const Self = @This();

    pub fn init(allocator: Allocator, dataset: *const Dataset, batch_size: usize, shuffle: bool) !Self {
        const indices = try allocator.alloc(usize, dataset.num_samples);
        for (indices, 0..) |*idx, i| {
            idx.* = i;
        }

        var loader = Self{
            .allocator = allocator,
            .dataset = dataset,
            .batch_size = batch_size,
            .shuffle = shuffle,
            .indices = indices,
            .current_idx = 0,
            .rng = std.Random.DefaultPrng.init(42),
        };

        if (shuffle) {
            loader.shuffleIndices();
        }

        return loader;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.indices);
        self.* = undefined;
    }

    fn shuffleIndices(self: *Self) void {
        const rand = self.rng.random();
        var i: usize = self.indices.len - 1;
        while (i > 0) : (i -= 1) {
            const j = rand.uintLessThan(usize, i + 1);
            const tmp = self.indices[i];
            self.indices[i] = self.indices[j];
            self.indices[j] = tmp;
        }
    }

    /// Get number of batches in one epoch
    pub fn numBatches(self: *const Self) usize {
        return (self.dataset.num_samples + self.batch_size - 1) / self.batch_size;
    }

    /// Get next batch. Returns null when epoch is complete.
    pub fn nextBatch(self: *Self) !?struct { inputs: Tensor, targets: Tensor } {
        if (self.current_idx >= self.dataset.num_samples) {
            return null;
        }

        const start = self.current_idx;
        const end = @min(start + self.batch_size, self.dataset.num_samples);
        const actual_batch_size = end - start;

        // Calculate shapes for batch
        const input_features = self.dataset.inputs.size() / self.dataset.num_samples;
        const target_features = self.dataset.targets.size() / self.dataset.num_samples;

        // Create batch tensors
        var batch_inputs = try Tensor.init(self.allocator, &[_]usize{ actual_batch_size, input_features });
        errdefer batch_inputs.deinit();

        var batch_targets = try Tensor.init(self.allocator, &[_]usize{ actual_batch_size, target_features });
        errdefer batch_targets.deinit();

        // Copy data
        for (0..actual_batch_size) |b| {
            const sample_idx = self.indices[start + b];

            // Copy input
            const in_start = sample_idx * input_features;
            for (0..input_features) |f| {
                batch_inputs.data[b * input_features + f] = self.dataset.inputs.data[in_start + f];
            }

            // Copy target
            const tgt_start = sample_idx * target_features;
            for (0..target_features) |f| {
                batch_targets.data[b * target_features + f] = self.dataset.targets.data[tgt_start + f];
            }
        }

        self.current_idx = end;

        return .{ .inputs = batch_inputs, .targets = batch_targets };
    }

    /// Reset for new epoch
    pub fn reset(self: *Self) void {
        self.current_idx = 0;
        if (self.shuffle) {
            self.shuffleIndices();
        }
    }
};

// Tests

test "dataset init" {
    const allocator = std.testing.allocator;

    var inputs = try Tensor.init(allocator, &[_]usize{ 10, 4 });
    defer inputs.deinit();
    inputs.fill(1.0);

    var targets = try Tensor.init(allocator, &[_]usize{ 10, 2 });
    defer targets.deinit();
    targets.fill(0.0);

    const dataset = try Dataset.init(inputs, targets);
    // Note: don't deinit dataset here since it takes ownership
    // and inputs/targets are deferred above

    try std.testing.expectEqual(@as(usize, 10), dataset.num_samples);
}

test "dataloader batches" {
    const allocator = std.testing.allocator;

    var inputs = try Tensor.init(allocator, &[_]usize{ 10, 4 });
    inputs.fill(1.0);

    var targets = try Tensor.init(allocator, &[_]usize{ 10, 2 });
    targets.fill(0.0);

    var dataset = try Dataset.init(inputs, targets);
    defer dataset.deinit();

    var loader = try DataLoader.init(allocator, &dataset, 3, false);
    defer loader.deinit();

    // Should have 4 batches (3 + 3 + 3 + 1)
    try std.testing.expectEqual(@as(usize, 4), loader.numBatches());

    var batch_count: usize = 0;
    var total_samples: usize = 0;

    while (try loader.nextBatch()) |batch| {
        var batch_inputs = batch.inputs;
        var batch_targets = batch.targets;
        defer batch_inputs.deinit();
        defer batch_targets.deinit();

        batch_count += 1;
        total_samples += batch_inputs.shape[0];
    }

    try std.testing.expectEqual(@as(usize, 4), batch_count);
    try std.testing.expectEqual(@as(usize, 10), total_samples);
}

test "dataloader reset" {
    const allocator = std.testing.allocator;

    var inputs = try Tensor.init(allocator, &[_]usize{ 6, 2 });
    inputs.fill(1.0);

    var targets = try Tensor.init(allocator, &[_]usize{ 6, 1 });
    targets.fill(0.0);

    var dataset = try Dataset.init(inputs, targets);
    defer dataset.deinit();

    var loader = try DataLoader.init(allocator, &dataset, 2, false);
    defer loader.deinit();

    // Consume all batches
    while (try loader.nextBatch()) |batch| {
        var b_in = batch.inputs;
        var b_tgt = batch.targets;
        b_in.deinit();
        b_tgt.deinit();
    }

    // Should return null
    const after_epoch = try loader.nextBatch();
    try std.testing.expect(after_epoch == null);

    // Reset and try again
    loader.reset();
    const after_reset = try loader.nextBatch();
    try std.testing.expect(after_reset != null);

    if (after_reset) |batch| {
        var b_in = batch.inputs;
        var b_tgt = batch.targets;
        b_in.deinit();
        b_tgt.deinit();
    }
}

test "dataloader shuffle" {
    const allocator = std.testing.allocator;

    var inputs = try Tensor.init(allocator, &[_]usize{ 100, 1 });
    for (0..100) |i| {
        inputs.data[i] = @floatFromInt(i);
    }

    const targets = try Tensor.zeros(allocator, &[_]usize{ 100, 1 });

    var dataset = try Dataset.init(inputs, targets);
    defer dataset.deinit();

    var loader = try DataLoader.init(allocator, &dataset, 100, true);
    defer loader.deinit();

    // Get first batch
    const batch = (try loader.nextBatch()).?;
    var batch_inputs = batch.inputs;
    var batch_targets = batch.targets;
    defer batch_inputs.deinit();
    defer batch_targets.deinit();

    // Check that it's not in order (very unlikely to be in order after shuffle)
    var in_order = true;
    for (0..100) |i| {
        if (batch_inputs.data[i] != @as(f32, @floatFromInt(i))) {
            in_order = false;
            break;
        }
    }
    try std.testing.expect(!in_order);
}
