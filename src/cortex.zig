/// Cortex - Neural Network Library
///
/// A minimalist neural network library with explicit forward and backward passes.
/// No autograd - every gradient is computed manually.

pub const tensor = @import("tensor/tensor.zig");
pub const ops = @import("tensor/ops.zig");
pub const matmul = @import("tensor/matmul.zig");
pub const reduce = @import("tensor/reduce.zig");
pub const broadcast = @import("tensor/broadcast.zig");
pub const dense = @import("layers/dense.zig");
pub const relu = @import("activations/relu.zig");
pub const sigmoid = @import("activations/sigmoid.zig");
pub const tanh_act = @import("activations/tanh.zig");
pub const softmax = @import("activations/softmax.zig");
pub const cross_entropy = @import("losses/cross_entropy.zig");
pub const mse = @import("losses/mse.zig");
pub const sgd = @import("optim/sgd.zig");
pub const adam = @import("optim/adam.zig");
pub const sequential = @import("model/sequential.zig");
pub const Tensor = tensor.Tensor;
pub const Dense = dense.Dense;
pub const ReLU = relu.ReLU;
pub const Sigmoid = sigmoid.Sigmoid;
pub const Tanh = tanh_act.Tanh;
pub const Softmax = softmax.Softmax;
pub const CrossEntropyLoss = cross_entropy.CrossEntropyLoss;
pub const MSELoss = mse.MSELoss;
pub const SGD = sgd.SGD;
pub const Adam = adam.Adam;
pub const Sequential = sequential.Sequential;
pub const Layer = sequential.Layer;

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
    _ = dense;
    _ = relu;
    _ = sigmoid;
    _ = tanh_act;
    _ = softmax;
    _ = cross_entropy;
    _ = mse;
    _ = sgd;
    _ = adam;
    _ = sequential;
}
