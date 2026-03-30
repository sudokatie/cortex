/// Cortex - Neural Network Library
///
/// A minimalist neural network library with explicit forward and backward passes.
/// No autograd - every gradient is computed manually.

pub const tensor = @import("tensor/tensor.zig");
pub const ops = @import("tensor/ops.zig");
pub const matmul = @import("tensor/matmul.zig");
pub const reduce = @import("tensor/reduce.zig");
pub const broadcast = @import("tensor/broadcast.zig");
pub const Tensor = tensor.Tensor;

// Module exports will be added as implemented:
// pub const layers = @import("layers/layer.zig");
// pub const activations = @import("activations/activation.zig");
// pub const losses = @import("losses/loss.zig");
// pub const optim = @import("optim/optimizer.zig");
// pub const data = @import("data/loader.zig");
// pub const model = @import("model/sequential.zig");

test {
    _ = tensor;
    _ = ops;
    _ = matmul;
    _ = reduce;
    _ = broadcast;
}
