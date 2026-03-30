const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Dataset = @import("loader.zig").Dataset;

/// MNIST dataset loader
/// Parses IDX file format used by MNIST
pub const MNIST = struct {
    allocator: Allocator,
    train_images: Tensor,
    train_labels: Tensor,
    test_images: Tensor,
    test_labels: Tensor,
    num_train: usize,
    num_test: usize,

    const Self = @This();

    /// Load MNIST from directory containing the 4 IDX files
    pub fn load(allocator: Allocator, dir_path: []const u8) !Self {
        // Build file paths
        var train_images_path: [512]u8 = undefined;
        var train_labels_path: [512]u8 = undefined;
        var test_images_path: [512]u8 = undefined;
        var test_labels_path: [512]u8 = undefined;

        const train_img = std.fmt.bufPrint(&train_images_path, "{s}/train-images-idx3-ubyte", .{dir_path}) catch return error.PathTooLong;
        const train_lbl = std.fmt.bufPrint(&train_labels_path, "{s}/train-labels-idx1-ubyte", .{dir_path}) catch return error.PathTooLong;
        const test_img = std.fmt.bufPrint(&test_images_path, "{s}/t10k-images-idx3-ubyte", .{dir_path}) catch return error.PathTooLong;
        const test_lbl = std.fmt.bufPrint(&test_labels_path, "{s}/t10k-labels-idx1-ubyte", .{dir_path}) catch return error.PathTooLong;

        // Load files
        const train_images = try loadImages(allocator, train_img);
        errdefer train_images.deinit();

        const train_labels = try loadLabels(allocator, train_lbl);
        errdefer train_labels.deinit();

        const test_images = try loadImages(allocator, test_img);
        errdefer test_images.deinit();

        const test_labels = try loadLabels(allocator, test_lbl);

        return Self{
            .allocator = allocator,
            .train_images = train_images,
            .train_labels = train_labels,
            .test_images = test_images,
            .test_labels = test_labels,
            .num_train = train_images.shape[0],
            .num_test = test_images.shape[0],
        };
    }

    pub fn deinit(self: *Self) void {
        self.train_images.deinit();
        self.train_labels.deinit();
        self.test_images.deinit();
        self.test_labels.deinit();
        self.* = undefined;
    }

    /// Get training dataset
    pub fn trainDataset(self: *const Self) !Dataset {
        const images = try self.train_images.clone();
        errdefer images.deinit();
        const labels = try self.train_labels.clone();
        return Dataset.init(images, labels);
    }

    /// Get test dataset
    pub fn testDataset(self: *const Self) !Dataset {
        const images = try self.test_images.clone();
        errdefer images.deinit();
        const labels = try self.test_labels.clone();
        return Dataset.init(images, labels);
    }
};

/// Read big-endian u32 from bytes
fn readBigEndianU32(bytes: []const u8) u32 {
    return (@as(u32, bytes[0]) << 24) |
        (@as(u32, bytes[1]) << 16) |
        (@as(u32, bytes[2]) << 8) |
        @as(u32, bytes[3]);
}

/// Load images from IDX3 format file
fn loadImages(allocator: Allocator, path: []const u8) !Tensor {
    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
    defer file.close();

    // Read header
    var header: [16]u8 = undefined;
    _ = try file.readAll(&header);

    const magic = readBigEndianU32(header[0..4]);
    if (magic != 0x00000803) {
        return error.InvalidMagicNumber;
    }

    const num_images = readBigEndianU32(header[4..8]);
    const rows = readBigEndianU32(header[8..12]);
    const cols = readBigEndianU32(header[12..16]);
    const image_size = rows * cols;

    // Read all image data
    const data_size = num_images * image_size;
    const raw_data = try allocator.alloc(u8, data_size);
    defer allocator.free(raw_data);

    _ = try file.readAll(raw_data);

    // Create tensor and normalize to [0, 1]
    const tensor = try Tensor.init(allocator, &[_]usize{ num_images, image_size });
    for (tensor.data, raw_data) |*t, r| {
        t.* = @as(f32, @floatFromInt(r)) / 255.0;
    }

    return tensor;
}

/// Load labels from IDX1 format file and convert to one-hot
fn loadLabels(allocator: Allocator, path: []const u8) !Tensor {
    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
    defer file.close();

    // Read header
    var header: [8]u8 = undefined;
    _ = try file.readAll(&header);

    const magic = readBigEndianU32(header[0..4]);
    if (magic != 0x00000801) {
        return error.InvalidMagicNumber;
    }

    const num_labels = readBigEndianU32(header[4..8]);

    // Read all label data
    const raw_labels = try allocator.alloc(u8, num_labels);
    defer allocator.free(raw_labels);

    _ = try file.readAll(raw_labels);

    // Create one-hot tensor [num_labels, 10]
    var tensor = try Tensor.zeros(allocator, &[_]usize{ num_labels, 10 });
    for (raw_labels, 0..) |label, i| {
        tensor.data[i * 10 + label] = 1.0;
    }

    return tensor;
}

/// Create synthetic MNIST-like data for testing
pub fn createSyntheticData(allocator: Allocator, num_samples: usize) !struct { images: Tensor, labels: Tensor } {
    var images = try Tensor.init(allocator, &[_]usize{ num_samples, 784 });
    errdefer images.deinit();

    var labels = try Tensor.zeros(allocator, &[_]usize{ num_samples, 10 });

    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    for (0..num_samples) |i| {
        // Random image (normalized)
        for (0..784) |j| {
            images.data[i * 784 + j] = rand.float(f32);
        }
        // Random label (one-hot)
        const label = rand.uintLessThan(usize, 10);
        labels.data[i * 10 + label] = 1.0;
    }

    return .{ .images = images, .labels = labels };
}

// Tests

test "synthetic mnist data" {
    const allocator = std.testing.allocator;

    const data = try createSyntheticData(allocator, 100);
    var images = data.images;
    var labels = data.labels;
    defer images.deinit();
    defer labels.deinit();

    // Check shapes
    try std.testing.expectEqual(@as(usize, 100), images.shape[0]);
    try std.testing.expectEqual(@as(usize, 784), images.shape[1]);
    try std.testing.expectEqual(@as(usize, 100), labels.shape[0]);
    try std.testing.expectEqual(@as(usize, 10), labels.shape[1]);
}

test "synthetic mnist labels one-hot" {
    const allocator = std.testing.allocator;

    const data = try createSyntheticData(allocator, 50);
    var images = data.images;
    var labels = data.labels;
    defer images.deinit();
    defer labels.deinit();

    // Each label row should sum to 1 (one-hot)
    for (0..50) |i| {
        var sum: f32 = 0.0;
        for (0..10) |j| {
            sum += labels.data[i * 10 + j];
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
    }
}

test "synthetic mnist normalized" {
    const allocator = std.testing.allocator;

    const data = try createSyntheticData(allocator, 20);
    var images = data.images;
    var labels = data.labels;
    defer images.deinit();
    defer labels.deinit();

    // All image values should be in [0, 1]
    for (images.data) |v| {
        try std.testing.expect(v >= 0.0);
        try std.testing.expect(v <= 1.0);
    }
}
