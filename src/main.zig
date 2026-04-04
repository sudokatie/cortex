const std = @import("std");
const cortex = @import("cortex.zig");

pub fn main() !void {
    var args = std.process.args();
    _ = args.skip(); // skip program name
    
    const command = args.next() orelse {
        printUsage();
        return;
    };
    
    if (std.mem.eql(u8, command, "train")) {
        std.debug.print("Training not yet implemented\n", .{});
    } else if (std.mem.eql(u8, command, "infer")) {
        std.debug.print("Inference not yet implemented\n", .{});
    } else if (std.mem.eql(u8, command, "info")) {
        printInfo();
    } else if (std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        printUsage();
    } else {
        std.debug.print("Unknown command: {s}\n", .{command});
        printUsage();
    }
}

fn printUsage() void {
    std.debug.print(
        \\cortex - Neural network library
        \\
        \\USAGE:
        \\    cortex <command> [options]
        \\
        \\COMMANDS:
        \\    train    Train a model on a dataset
        \\    infer    Run inference on input data
        \\    info     Show version and build info
        \\
        \\OPTIONS:
        \\    -h, --help    Show this help message
        \\
    , .{});
}

fn printInfo() void {
    std.debug.print(
        \\cortex v0.1.0
        \\Neural network library from scratch
        \\
        \\Features:
        \\  - Tensor operations
        \\  - Dense, Conv2D, LSTM layers
        \\  - ReLU, Sigmoid, Tanh, Softmax activations
        \\  - Cross-entropy, MSE losses
        \\  - SGD, Adam optimizers
        \\  - MNIST, CIFAR-10 data loaders
        \\
    , .{});
}

test "main compiles" {
    // Just verify compilation
}
