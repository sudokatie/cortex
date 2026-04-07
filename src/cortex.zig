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
pub const conv2d = @import("layers/conv2d.zig");
pub const pool = @import("layers/pool.zig");
pub const relu = @import("activations/relu.zig");
pub const sigmoid = @import("activations/sigmoid.zig");
pub const tanh_act = @import("activations/tanh.zig");
pub const softmax = @import("activations/softmax.zig");
pub const cross_entropy = @import("losses/cross_entropy.zig");
pub const mse = @import("losses/mse.zig");
pub const sgd = @import("optim/sgd.zig");
pub const adam = @import("optim/adam.zig");
pub const sequential = @import("model/sequential.zig");
pub const serialize = @import("model/serialize.zig");
pub const random = @import("util/random.zig");
pub const loader = @import("data/loader.zig");
pub const mnist = @import("data/mnist.zig");
pub const Tensor = tensor.Tensor;
pub const Dense = dense.Dense;
pub const Conv2D = conv2d.Conv2D;
pub const MaxPool2D = pool.MaxPool2D;
pub const AvgPool2D = pool.AvgPool2D;
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
pub const save = serialize.save;
pub const load = serialize.load;
pub const saveToBytes = serialize.saveToBytes;
pub const loadFromBytes = serialize.loadFromBytes;
pub const Random = random.Random;
pub const WeightInit = random.WeightInit;
pub const Dataset = loader.Dataset;
pub const DataLoader = loader.DataLoader;
pub const MNIST = mnist.MNIST;

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
    _ = conv2d;
    _ = pool;
    _ = relu;
    _ = sigmoid;
    _ = tanh_act;
    _ = softmax;
    _ = cross_entropy;
    _ = mse;
    _ = sgd;
    _ = adam;
    _ = sequential;
    _ = serialize;
    _ = random;
    _ = loader;
    _ = mnist;
}
