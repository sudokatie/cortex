const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Dense = @import("../layers/dense.zig").Dense;
const ReLU = @import("../activations/relu.zig").ReLU;
const Sigmoid = @import("../activations/sigmoid.zig").Sigmoid;
const Tanh = @import("../activations/tanh.zig").Tanh;
const Sequential = @import("sequential.zig").Sequential;
const Layer = @import("sequential.zig").Layer;

/// Magic bytes for Cortex model files
const MAGIC: [4]u8 = .{ 'C', 'R', 'T', 'X' };

/// Current file format version
const VERSION: u32 = 1;

/// Layer type identifiers
const LayerType = enum(u8) {
    dense = 0,
    relu = 1,
    sigmoid = 2,
    tanh = 3,
};

/// Error types for serialization
pub const SerializeError = error{
    InvalidMagic,
    UnsupportedVersion,
    InvalidLayerType,
    ReadError,
    WriteError,
    CorruptedFile,
    OutOfMemory,
};

/// Save a Sequential model to a file
pub fn save(model: *const Sequential, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var writer = file.writer();

    // Write header
    try writer.writeAll(&MAGIC);
    try writer.writeInt(u32, VERSION, .little);
    try writer.writeInt(u32, @intCast(model.layers.items.len), .little);

    // Write each layer
    for (model.layers.items) |layer| {
        try writeLayer(&writer, layer);
    }
}

/// Save a Sequential model to a byte buffer
pub fn saveToBytes(allocator: Allocator, model: *const Sequential) ![]u8 {
    // Calculate size first
    const size = calculateSize(model);

    // Allocate buffer
    const buffer = try allocator.alloc(u8, size);
    errdefer allocator.free(buffer);

    // Write to buffer
    var fbs = std.io.fixedBufferStream(buffer);
    var writer = fbs.writer();

    // Write header
    try writer.writeAll(&MAGIC);
    try writer.writeInt(u32, VERSION, .little);
    try writer.writeInt(u32, @intCast(model.layers.items.len), .little);

    // Write each layer
    for (model.layers.items) |layer| {
        try writeLayer(&writer, layer);
    }

    return buffer;
}

/// Load a Sequential model from a file
pub fn load(allocator: Allocator, path: []const u8) !Sequential {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var reader = file.reader();
    return loadFromReader(allocator, &reader);
}

/// Load a Sequential model from a byte buffer
pub fn loadFromBytes(allocator: Allocator, bytes: []const u8) !Sequential {
    var fbs = std.io.fixedBufferStream(bytes);
    var reader = fbs.reader();
    return loadFromReader(allocator, &reader);
}

fn loadFromReader(allocator: Allocator, reader: anytype) !Sequential {
    // Read and verify magic
    var magic: [4]u8 = undefined;
    const bytes_read = try reader.readAll(&magic);
    if (bytes_read != 4 or !std.mem.eql(u8, &magic, &MAGIC)) {
        return SerializeError.InvalidMagic;
    }

    // Read and verify version
    const version = try reader.readInt(u32, .little);
    if (version != VERSION) {
        return SerializeError.UnsupportedVersion;
    }

    // Read layer count
    const layer_count = try reader.readInt(u32, .little);

    // Create model
    var model = Sequential.init(allocator);
    errdefer model.deinit();

    // Track dense layers for cleanup on error
    var dense_layers: std.ArrayListUnmanaged(*Dense) = .{};
    defer {
        for (dense_layers.items) |d| {
            d.deinit();
            allocator.destroy(d);
        }
        dense_layers.deinit(allocator);
    }

    // Track activation layers
    var relu_layers: std.ArrayListUnmanaged(*ReLU) = .{};
    defer {
        for (relu_layers.items) |r| {
            r.deinit();
            allocator.destroy(r);
        }
        relu_layers.deinit(allocator);
    }

    var sigmoid_layers: std.ArrayListUnmanaged(*Sigmoid) = .{};
    defer {
        for (sigmoid_layers.items) |s| {
            s.deinit();
            allocator.destroy(s);
        }
        sigmoid_layers.deinit(allocator);
    }

    var tanh_layers: std.ArrayListUnmanaged(*Tanh) = .{};
    defer {
        for (tanh_layers.items) |t| {
            t.deinit();
            allocator.destroy(t);
        }
        tanh_layers.deinit(allocator);
    }

    // Read layers
    for (0..layer_count) |_| {
        const layer_type_byte = try reader.readByte();
        const layer_type = std.meta.intToEnum(LayerType, layer_type_byte) catch {
            return SerializeError.InvalidLayerType;
        };

        switch (layer_type) {
            .dense => {
                const dense = try readDense(allocator, reader);
                try dense_layers.append(allocator, dense);
                try model.addDense(dense);
            },
            .relu => {
                const relu = try allocator.create(ReLU);
                relu.* = ReLU.init(allocator);
                try relu_layers.append(allocator, relu);
                try model.addReLU(relu);
            },
            .sigmoid => {
                const sigmoid = try allocator.create(Sigmoid);
                sigmoid.* = Sigmoid.init(allocator);
                try sigmoid_layers.append(allocator, sigmoid);
                try model.addSigmoid(sigmoid);
            },
            .tanh => {
                const tanh_layer = try allocator.create(Tanh);
                tanh_layer.* = Tanh.init(allocator);
                try tanh_layers.append(allocator, tanh_layer);
                try model.addTanh(tanh_layer);
            },
        }
    }

    // Clear tracking lists without freeing (ownership transferred to caller)
    dense_layers.clearRetainingCapacity();
    relu_layers.clearRetainingCapacity();
    sigmoid_layers.clearRetainingCapacity();
    tanh_layers.clearRetainingCapacity();

    return model;
}

fn writeLayer(writer: anytype, layer: Layer) !void {
    switch (layer) {
        .dense => |d| {
            try writer.writeByte(@intFromEnum(LayerType.dense));
            try writeDense(writer, d);
        },
        .relu => {
            try writer.writeByte(@intFromEnum(LayerType.relu));
            // Stateless - no data to write
        },
        .sigmoid => {
            try writer.writeByte(@intFromEnum(LayerType.sigmoid));
            // Stateless - no data to write
        },
        .tanh => {
            try writer.writeByte(@intFromEnum(LayerType.tanh));
            // Stateless - no data to write
        },
    }
}

fn writeDense(writer: anytype, dense: *const Dense) !void {
    // Write dimensions
    const in_features = dense.weights.shape[0];
    const out_features = dense.weights.shape[1];
    try writer.writeInt(u32, @intCast(in_features), .little);
    try writer.writeInt(u32, @intCast(out_features), .little);

    // Write weights as raw f32 bytes
    const weights_bytes = std.mem.sliceAsBytes(dense.weights.data);
    try writer.writeAll(weights_bytes);

    // Write bias as raw f32 bytes
    const bias_bytes = std.mem.sliceAsBytes(dense.bias.data);
    try writer.writeAll(bias_bytes);
}

fn readDense(allocator: Allocator, reader: anytype) !*Dense {
    // Read dimensions
    const in_features = try reader.readInt(u32, .little);
    const out_features = try reader.readInt(u32, .little);

    // Create dense layer (initializes with random weights)
    const dense = try allocator.create(Dense);
    errdefer allocator.destroy(dense);

    dense.* = try Dense.init(allocator, in_features, out_features);
    errdefer dense.deinit();

    // Read weights
    const weights_bytes = std.mem.sliceAsBytes(dense.weights.data);
    const weights_read = try reader.readAll(weights_bytes);
    if (weights_read != weights_bytes.len) {
        return SerializeError.CorruptedFile;
    }

    // Read bias
    const bias_bytes = std.mem.sliceAsBytes(dense.bias.data);
    const bias_read = try reader.readAll(bias_bytes);
    if (bias_read != bias_bytes.len) {
        return SerializeError.CorruptedFile;
    }

    return dense;
}

/// Get the size in bytes of a serialized model (without actually writing)
pub fn calculateSize(model: *const Sequential) usize {
    var size: usize = 4 + 4 + 4; // magic + version + layer_count

    for (model.layers.items) |layer| {
        size += 1; // layer type byte
        switch (layer) {
            .dense => |d| {
                size += 4 + 4; // in_features + out_features
                size += d.weights.data.len * @sizeOf(f32); // weights
                size += d.bias.data.len * @sizeOf(f32); // bias
            },
            .relu, .sigmoid, .tanh => {
                // Stateless - no additional data
            },
        }
    }

    return size;
}

// Tests

test "serialize empty model" {
    const allocator = std.testing.allocator;

    var model = Sequential.init(allocator);
    defer model.deinit();

    const bytes = try saveToBytes(allocator, &model);
    defer allocator.free(bytes);

    // Verify header
    try std.testing.expectEqualSlices(u8, &MAGIC, bytes[0..4]);
    try std.testing.expectEqual(@as(u32, VERSION), std.mem.readInt(u32, bytes[4..8], .little));
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, bytes[8..12], .little));
}

test "serialize and load empty model" {
    const allocator = std.testing.allocator;

    var model = Sequential.init(allocator);
    defer model.deinit();

    const bytes = try saveToBytes(allocator, &model);
    defer allocator.free(bytes);

    var loaded = try loadFromBytes(allocator, bytes);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 0), loaded.numLayers());
}

test "serialize dense layer" {
    const allocator = std.testing.allocator;

    var dense = try Dense.init(allocator, 4, 2);
    defer dense.deinit();

    // Set known values
    dense.weights.data[0] = 1.0;
    dense.weights.data[1] = 2.0;
    dense.bias.data[0] = 0.5;
    dense.bias.data[1] = 0.5;

    var model = Sequential.init(allocator);
    defer model.deinit();
    try model.addDense(&dense);

    const bytes = try saveToBytes(allocator, &model);
    defer allocator.free(bytes);

    // Expected: magic(4) + version(4) + count(4) + type(1) + in(4) + out(4) + weights(32) + bias(8)
    try std.testing.expectEqual(@as(usize, 61), bytes.len);
}

test "round trip dense relu dense model" {
    const allocator = std.testing.allocator;

    // Create original model
    var dense1 = try Dense.init(allocator, 4, 8);
    defer dense1.deinit();
    dense1.weights.data[0] = 1.5;
    dense1.weights.data[1] = -2.5;
    dense1.bias.data[0] = 0.1;

    var relu = ReLU.init(allocator);
    defer relu.deinit();

    var dense2 = try Dense.init(allocator, 8, 2);
    defer dense2.deinit();
    dense2.weights.data[0] = 3.14;
    dense2.bias.data[0] = -0.5;

    var model = Sequential.init(allocator);
    defer model.deinit();
    try model.addDense(&dense1);
    try model.addReLU(&relu);
    try model.addDense(&dense2);

    // Save and load
    const bytes = try saveToBytes(allocator, &model);
    defer allocator.free(bytes);

    var loaded = try loadFromBytes(allocator, bytes);
    defer {
        // Need to free the layers that were created during load
        for (loaded.layers.items) |layer| {
            switch (layer) {
                .dense => |d| {
                    d.deinit();
                    allocator.destroy(d);
                },
                .relu => |r| {
                    r.deinit();
                    allocator.destroy(r);
                },
                .sigmoid => |s| {
                    s.deinit();
                    allocator.destroy(s);
                },
                .tanh => |t| {
                    t.deinit();
                    allocator.destroy(t);
                },
            }
        }
        loaded.deinit();
    }

    // Verify structure
    try std.testing.expectEqual(@as(usize, 3), loaded.numLayers());

    // Verify weights preserved
    const loaded_dense1 = loaded.layers.items[0].dense;
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), loaded_dense1.weights.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), loaded_dense1.weights.data[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), loaded_dense1.bias.data[0], 0.0001);

    const loaded_dense2 = loaded.layers.items[2].dense;
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), loaded_dense2.weights.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), loaded_dense2.bias.data[0], 0.0001);
}

test "calculate size matches actual" {
    const allocator = std.testing.allocator;

    var dense = try Dense.init(allocator, 10, 5);
    defer dense.deinit();

    var relu = ReLU.init(allocator);
    defer relu.deinit();

    var model = Sequential.init(allocator);
    defer model.deinit();
    try model.addDense(&dense);
    try model.addReLU(&relu);

    const calculated = calculateSize(&model);
    const bytes = try saveToBytes(allocator, &model);
    defer allocator.free(bytes);

    try std.testing.expectEqual(calculated, bytes.len);
}

test "invalid magic returns error" {
    const allocator = std.testing.allocator;
    const bad_bytes = [_]u8{ 'B', 'A', 'D', '!' } ++ [_]u8{0} ** 8;

    const result = loadFromBytes(allocator, &bad_bytes);
    try std.testing.expectError(SerializeError.InvalidMagic, result);
}

test "unsupported version returns error" {
    const allocator = std.testing.allocator;
    var bad_bytes: [12]u8 = undefined;
    @memcpy(bad_bytes[0..4], &MAGIC);
    std.mem.writeInt(u32, bad_bytes[4..8], 999, .little); // bad version
    std.mem.writeInt(u32, bad_bytes[8..12], 0, .little);

    const result = loadFromBytes(allocator, &bad_bytes);
    try std.testing.expectError(SerializeError.UnsupportedVersion, result);
}

test "invalid layer type returns error" {
    const allocator = std.testing.allocator;
    var bad_bytes: [13]u8 = undefined;
    @memcpy(bad_bytes[0..4], &MAGIC);
    std.mem.writeInt(u32, bad_bytes[4..8], VERSION, .little);
    std.mem.writeInt(u32, bad_bytes[8..12], 1, .little); // 1 layer
    bad_bytes[12] = 255; // invalid layer type

    const result = loadFromBytes(allocator, &bad_bytes);
    try std.testing.expectError(SerializeError.InvalidLayerType, result);
}

test "round trip with sigmoid and tanh" {
    const allocator = std.testing.allocator;

    var dense1 = try Dense.init(allocator, 2, 4);
    defer dense1.deinit();

    var sigmoid = Sigmoid.init(allocator);
    defer sigmoid.deinit();

    var dense2 = try Dense.init(allocator, 4, 2);
    defer dense2.deinit();

    var tanh_layer = Tanh.init(allocator);
    defer tanh_layer.deinit();

    var model = Sequential.init(allocator);
    defer model.deinit();
    try model.addDense(&dense1);
    try model.addSigmoid(&sigmoid);
    try model.addDense(&dense2);
    try model.addTanh(&tanh_layer);

    const bytes = try saveToBytes(allocator, &model);
    defer allocator.free(bytes);

    var loaded = try loadFromBytes(allocator, bytes);
    defer {
        for (loaded.layers.items) |layer| {
            switch (layer) {
                .dense => |d| {
                    d.deinit();
                    allocator.destroy(d);
                },
                .relu => |r| {
                    r.deinit();
                    allocator.destroy(r);
                },
                .sigmoid => |s| {
                    s.deinit();
                    allocator.destroy(s);
                },
                .tanh => |t| {
                    t.deinit();
                    allocator.destroy(t);
                },
            }
        }
        loaded.deinit();
    }

    try std.testing.expectEqual(@as(usize, 4), loaded.numLayers());
}

test "forward pass after load produces same output" {
    const allocator = std.testing.allocator;

    // Create model with known weights
    var dense = try Dense.init(allocator, 2, 2);
    defer dense.deinit();
    dense.weights.data[0] = 1.0;
    dense.weights.data[1] = 0.0;
    dense.weights.data[2] = 0.0;
    dense.weights.data[3] = 1.0;
    dense.bias.data[0] = 0.5;
    dense.bias.data[1] = -0.5;

    var model = Sequential.init(allocator);
    defer model.deinit();
    try model.addDense(&dense);

    // Get original output
    var x = try Tensor.init(allocator, &[_]usize{ 1, 2 });
    defer x.deinit();
    x.data[0] = 1.0;
    x.data[1] = 2.0;

    var y_original = try model.forward(&x);
    defer y_original.deinit();

    // Save and load
    const bytes = try saveToBytes(allocator, &model);
    defer allocator.free(bytes);

    var loaded = try loadFromBytes(allocator, bytes);
    defer {
        for (loaded.layers.items) |layer| {
            switch (layer) {
                .dense => |d| {
                    d.deinit();
                    allocator.destroy(d);
                },
                .relu => |r| {
                    r.deinit();
                    allocator.destroy(r);
                },
                .sigmoid => |s| {
                    s.deinit();
                    allocator.destroy(s);
                },
                .tanh => |t| {
                    t.deinit();
                    allocator.destroy(t);
                },
            }
        }
        loaded.deinit();
    }

    // Get loaded output
    var y_loaded = try loaded.forward(&x);
    defer y_loaded.deinit();

    // Verify outputs match
    try std.testing.expectApproxEqAbs(y_original.data[0], y_loaded.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(y_original.data[1], y_loaded.data[1], 0.0001);
}
