const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const matmul = @import("../tensor/matmul.zig");

/// Scaled Dot-Product Attention.
/// Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
///
/// For self-attention, Q = K = V = input @ W_q/W_k/W_v
pub const SelfAttention = struct {
    /// Query projection weights [embed_dim, embed_dim]
    w_query: Tensor,
    /// Key projection weights [embed_dim, embed_dim]
    w_key: Tensor,
    /// Value projection weights [embed_dim, embed_dim]
    w_value: Tensor,
    /// Output projection weights [embed_dim, embed_dim]
    w_output: Tensor,
    /// Embedding dimension
    embed_dim: usize,
    /// Scaling factor (1 / sqrt(embed_dim))
    scale: f32,
    /// Cached values for backward pass
    cached_input: ?Tensor,
    cached_query: ?Tensor,
    cached_key: ?Tensor,
    cached_value: ?Tensor,
    cached_attention: ?Tensor,
    /// Allocator
    allocator: Allocator,

    const Self = @This();

    /// Create a new self-attention layer.
    pub fn init(allocator: Allocator, embed_dim: usize) !Self {
        // Xavier initialization scale
        const init_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(embed_dim)));

        var prng = std.Random.DefaultPrng.init(42);
        const rand = prng.random();

        var w_query = try Tensor.init(allocator, &[_]usize{ embed_dim, embed_dim });
        errdefer w_query.deinit();
        for (w_query.data) |*w| {
            w.* = (rand.float(f32) * 2.0 - 1.0) * init_scale;
        }

        var w_key = try Tensor.init(allocator, &[_]usize{ embed_dim, embed_dim });
        errdefer w_key.deinit();
        for (w_key.data) |*w| {
            w.* = (rand.float(f32) * 2.0 - 1.0) * init_scale;
        }

        var w_value = try Tensor.init(allocator, &[_]usize{ embed_dim, embed_dim });
        errdefer w_value.deinit();
        for (w_value.data) |*w| {
            w.* = (rand.float(f32) * 2.0 - 1.0) * init_scale;
        }

        var w_output = try Tensor.init(allocator, &[_]usize{ embed_dim, embed_dim });
        errdefer w_output.deinit();
        for (w_output.data) |*w| {
            w.* = (rand.float(f32) * 2.0 - 1.0) * init_scale;
        }

        return Self{
            .w_query = w_query,
            .w_key = w_key,
            .w_value = w_value,
            .w_output = w_output,
            .embed_dim = embed_dim,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(embed_dim))),
            .cached_input = null,
            .cached_query = null,
            .cached_key = null,
            .cached_value = null,
            .cached_attention = null,
            .allocator = allocator,
        };
    }

    /// Free all memory.
    pub fn deinit(self: *Self) void {
        self.w_query.deinit();
        self.w_key.deinit();
        self.w_value.deinit();
        self.w_output.deinit();
        if (self.cached_input) |*t| t.deinit();
        if (self.cached_query) |*t| t.deinit();
        if (self.cached_key) |*t| t.deinit();
        if (self.cached_value) |*t| t.deinit();
        if (self.cached_attention) |*t| t.deinit();
        self.* = undefined;
    }

    /// Forward pass.
    /// Input: [batch, seq_len, embed_dim]
    /// Output: [batch, seq_len, embed_dim]
    ///
    /// For simplicity, this implementation handles [seq_len, embed_dim] (no batch dim)
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        // Clear cached values
        if (self.cached_input) |*t| t.deinit();
        if (self.cached_query) |*t| t.deinit();
        if (self.cached_key) |*t| t.deinit();
        if (self.cached_value) |*t| t.deinit();
        if (self.cached_attention) |*t| t.deinit();

        const seq_len = x.shape[0];
        const embed_dim = x.shape[1];

        if (embed_dim != self.embed_dim) {
            return error.DimensionMismatch;
        }

        // Q = X @ W_q [seq_len, embed_dim]
        var query = try matmul.matmul(self.allocator, x, &self.w_query);
        errdefer query.deinit();

        // K = X @ W_k [seq_len, embed_dim]
        var key = try matmul.matmul(self.allocator, x, &self.w_key);
        errdefer key.deinit();

        // V = X @ W_v [seq_len, embed_dim]
        var value = try matmul.matmul(self.allocator, x, &self.w_value);
        errdefer value.deinit();

        // Attention scores = Q @ K.T / sqrt(d_k) [seq_len, seq_len]
        var scores = try matmul.matmulTransposeB(self.allocator, &query, &key);
        errdefer scores.deinit();

        // Scale
        for (scores.data) |*s| {
            s.* *= self.scale;
        }

        // Softmax over last dimension (each row)
        var attention = try Tensor.init(self.allocator, scores.shape);
        errdefer attention.deinit();

        for (0..seq_len) |i| {
            // Find max for numerical stability
            var max_val: f32 = scores.data[i * seq_len];
            for (0..seq_len) |j| {
                const val = scores.data[i * seq_len + j];
                if (val > max_val) max_val = val;
            }

            // Compute exp and sum
            var sum: f32 = 0.0;
            for (0..seq_len) |j| {
                const idx = i * seq_len + j;
                const exp_val = @exp(scores.data[idx] - max_val);
                attention.data[idx] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for (0..seq_len) |j| {
                attention.data[i * seq_len + j] /= sum;
            }
        }

        scores.deinit();

        // Output = attention @ V [seq_len, embed_dim]
        var attended = try matmul.matmul(self.allocator, &attention, &value);
        errdefer attended.deinit();

        // Final projection
        var output = try matmul.matmul(self.allocator, &attended, &self.w_output);
        errdefer output.deinit();

        attended.deinit();

        // Cache for backward
        self.cached_input = try x.clone();
        self.cached_query = query;
        self.cached_key = key;
        self.cached_value = value;
        self.cached_attention = attention;

        return output;
    }

    /// Backward pass.
    /// grad_output: [seq_len, embed_dim]
    /// Returns gradients for weights (grad_wq, grad_wk, grad_wv, grad_wo) and input
    pub fn backward(
        self: *Self,
        grad_output: *const Tensor,
        grad_w_query: *Tensor,
        grad_w_key: *Tensor,
        grad_w_value: *Tensor,
        grad_w_output: *Tensor,
    ) !Tensor {
        const input = &(self.cached_input orelse return error.NoCachedInput);
        const query = &(self.cached_query orelse return error.NoCachedQuery);
        const key = &(self.cached_key orelse return error.NoCachedKey);
        const value = &(self.cached_value orelse return error.NoCachedValue);
        const attention = &(self.cached_attention orelse return error.NoCachedAttention);

        const seq_len = input.shape[0];
        const embed_dim = input.shape[1];

        // Gradient through output projection
        // d_attended = grad_output @ W_o.T
        var grad_attended = try matmul.matmulTransposeB(self.allocator, grad_output, &self.w_output);
        defer grad_attended.deinit();

        // grad_W_o = attended.T @ grad_output (need to reconstruct attended)
        var attended = try matmul.matmul(self.allocator, attention, value);
        defer attended.deinit();

        var dwo = try matmul.matmulTransposeA(self.allocator, &attended, grad_output);
        defer dwo.deinit();
        @memcpy(grad_w_output.data, dwo.data);

        // Gradient through attention @ V
        // d_attention = grad_attended @ V.T [seq_len, seq_len]
        var grad_attention = try matmul.matmulTransposeB(self.allocator, &grad_attended, value);
        defer grad_attention.deinit();

        // d_V = attention.T @ grad_attended [seq_len, embed_dim]
        var grad_value = try matmul.matmulTransposeA(self.allocator, attention, &grad_attended);
        defer grad_value.deinit();

        // Gradient through softmax
        // For each row i: d_score[i] = attention[i] * (d_attention[i] - sum(d_attention[i] * attention[i]))
        var grad_scores = try Tensor.init(self.allocator, grad_attention.shape);
        defer grad_scores.deinit();

        for (0..seq_len) |i| {
            var dot: f32 = 0.0;
            for (0..seq_len) |j| {
                const idx = i * seq_len + j;
                dot += grad_attention.data[idx] * attention.data[idx];
            }
            for (0..seq_len) |j| {
                const idx = i * seq_len + j;
                grad_scores.data[idx] = attention.data[idx] * (grad_attention.data[idx] - dot);
            }
        }

        // Scale gradient
        for (grad_scores.data) |*g| {
            g.* *= self.scale;
        }

        // Gradient through Q @ K.T
        // d_Q = grad_scores @ K [seq_len, embed_dim]
        var grad_query = try matmul.matmul(self.allocator, &grad_scores, key);
        defer grad_query.deinit();

        // d_K = grad_scores.T @ Q [seq_len, embed_dim]
        var grad_key = try matmul.matmulTransposeA(self.allocator, &grad_scores, query);
        defer grad_key.deinit();

        // Gradient through projections
        // grad_W_q = input.T @ grad_query
        var dwq = try matmul.matmulTransposeA(self.allocator, input, &grad_query);
        defer dwq.deinit();
        @memcpy(grad_w_query.data, dwq.data);

        // grad_W_k = input.T @ grad_key
        var dwk = try matmul.matmulTransposeA(self.allocator, input, &grad_key);
        defer dwk.deinit();
        @memcpy(grad_w_key.data, dwk.data);

        // grad_W_v = input.T @ grad_value
        var dwv = try matmul.matmulTransposeA(self.allocator, input, &grad_value);
        defer dwv.deinit();
        @memcpy(grad_w_value.data, dwv.data);

        // Gradient through input
        // d_input = grad_query @ W_q.T + grad_key @ W_k.T + grad_value @ W_v.T
        var grad_input_q = try matmul.matmulTransposeB(self.allocator, &grad_query, &self.w_query);
        defer grad_input_q.deinit();

        var grad_input_k = try matmul.matmulTransposeB(self.allocator, &grad_key, &self.w_key);
        defer grad_input_k.deinit();

        var grad_input_v = try matmul.matmulTransposeB(self.allocator, &grad_value, &self.w_value);
        defer grad_input_v.deinit();

        var grad_input = try Tensor.zeros(self.allocator, &[_]usize{ seq_len, embed_dim });
        errdefer grad_input.deinit();

        for (0..grad_input.data.len) |i| {
            grad_input.data[i] = grad_input_q.data[i] + grad_input_k.data[i] + grad_input_v.data[i];
        }

        return grad_input;
    }
};

/// Multi-Head Attention.
/// Runs multiple attention heads in parallel and concatenates results.
pub const MultiHeadAttention = struct {
    /// Number of attention heads
    num_heads: usize,
    /// Embedding dimension (must be divisible by num_heads)
    embed_dim: usize,
    /// Head dimension
    head_dim: usize,
    /// Combined Q, K, V projection [embed_dim, 3 * embed_dim]
    w_qkv: Tensor,
    /// Output projection [embed_dim, embed_dim]
    w_output: Tensor,
    /// Scaling factor
    scale: f32,
    /// Cached values
    cached_input: ?Tensor,
    cached_attention: ?Tensor,
    /// Allocator
    allocator: Allocator,

    const Self = @This();

    /// Create multi-head attention.
    pub fn init(allocator: Allocator, embed_dim: usize, num_heads: usize) !Self {
        if (embed_dim % num_heads != 0) {
            return error.InvalidHeadCount;
        }

        const head_dim = embed_dim / num_heads;
        const init_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(embed_dim)));

        var prng = std.Random.DefaultPrng.init(42);
        const rand = prng.random();

        var w_qkv = try Tensor.init(allocator, &[_]usize{ embed_dim, 3 * embed_dim });
        errdefer w_qkv.deinit();
        for (w_qkv.data) |*w| {
            w.* = (rand.float(f32) * 2.0 - 1.0) * init_scale;
        }

        var w_output = try Tensor.init(allocator, &[_]usize{ embed_dim, embed_dim });
        errdefer w_output.deinit();
        for (w_output.data) |*w| {
            w.* = (rand.float(f32) * 2.0 - 1.0) * init_scale;
        }

        return Self{
            .num_heads = num_heads,
            .embed_dim = embed_dim,
            .head_dim = head_dim,
            .w_qkv = w_qkv,
            .w_output = w_output,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .cached_input = null,
            .cached_attention = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.w_qkv.deinit();
        self.w_output.deinit();
        if (self.cached_input) |*t| t.deinit();
        if (self.cached_attention) |*t| t.deinit();
        self.* = undefined;
    }

    /// Forward pass.
    /// Input: [seq_len, embed_dim]
    /// Output: [seq_len, embed_dim]
    pub fn forward(self: *Self, x: *const Tensor) !Tensor {
        if (self.cached_input) |*t| t.deinit();
        if (self.cached_attention) |*t| t.deinit();

        const seq_len = x.shape[0];
        const embed_dim = x.shape[1];

        if (embed_dim != self.embed_dim) {
            return error.DimensionMismatch;
        }

        // Project to Q, K, V in one operation
        // qkv = x @ W_qkv [seq_len, 3 * embed_dim]
        var qkv = try matmul.matmul(self.allocator, x, &self.w_qkv);
        defer qkv.deinit();

        // Split into Q, K, V and reshape for multi-head
        // For simplicity, process each head separately
        var output = try Tensor.zeros(self.allocator, &[_]usize{ seq_len, embed_dim });
        errdefer output.deinit();

        // Store attention for potential backward pass
        var all_attention = try Tensor.init(self.allocator, &[_]usize{ self.num_heads, seq_len, seq_len });
        errdefer all_attention.deinit();

        for (0..self.num_heads) |h| {
            const q_offset = h * self.head_dim;
            const k_offset = embed_dim + h * self.head_dim;
            const v_offset = 2 * embed_dim + h * self.head_dim;

            // Extract Q, K, V for this head
            var q_head = try Tensor.init(self.allocator, &[_]usize{ seq_len, self.head_dim });
            defer q_head.deinit();
            var k_head = try Tensor.init(self.allocator, &[_]usize{ seq_len, self.head_dim });
            defer k_head.deinit();
            var v_head = try Tensor.init(self.allocator, &[_]usize{ seq_len, self.head_dim });
            defer v_head.deinit();

            for (0..seq_len) |s| {
                for (0..self.head_dim) |d| {
                    q_head.data[s * self.head_dim + d] = qkv.data[s * (3 * embed_dim) + q_offset + d];
                    k_head.data[s * self.head_dim + d] = qkv.data[s * (3 * embed_dim) + k_offset + d];
                    v_head.data[s * self.head_dim + d] = qkv.data[s * (3 * embed_dim) + v_offset + d];
                }
            }

            // Compute attention for this head
            var scores = try matmul.matmulTransposeB(self.allocator, &q_head, &k_head);
            defer scores.deinit();

            // Scale and softmax
            for (scores.data) |*s| {
                s.* *= self.scale;
            }

            for (0..seq_len) |i| {
                var max_val: f32 = scores.data[i * seq_len];
                for (0..seq_len) |j| {
                    const val = scores.data[i * seq_len + j];
                    if (val > max_val) max_val = val;
                }

                var sum: f32 = 0.0;
                for (0..seq_len) |j| {
                    const idx = i * seq_len + j;
                    const exp_val = @exp(scores.data[idx] - max_val);
                    scores.data[idx] = exp_val;
                    sum += exp_val;
                }

                for (0..seq_len) |j| {
                    const idx = i * seq_len + j;
                    scores.data[idx] /= sum;
                    // Store in all_attention
                    all_attention.data[h * seq_len * seq_len + idx] = scores.data[idx];
                }
            }

            // attention @ V
            var head_output = try matmul.matmul(self.allocator, &scores, &v_head);
            defer head_output.deinit();

            // Add to output at correct position
            for (0..seq_len) |s| {
                for (0..self.head_dim) |d| {
                    output.data[s * embed_dim + h * self.head_dim + d] += head_output.data[s * self.head_dim + d];
                }
            }
        }

        // Final projection
        const final_output = try matmul.matmul(self.allocator, &output, &self.w_output);

        output.deinit();

        self.cached_input = try x.clone();
        self.cached_attention = all_attention;

        return final_output;
    }
};

// Tests

test "self attention forward shape" {
    const allocator = std.testing.allocator;

    var attn = try SelfAttention.init(allocator, 8);
    defer attn.deinit();

    // Input: [4, 8] (seq_len=4, embed_dim=8)
    var x = try Tensor.ones(allocator, &[_]usize{ 4, 8 });
    defer x.deinit();

    var y = try attn.forward(&x);
    defer y.deinit();

    try std.testing.expectEqual(@as(usize, 4), y.shape[0]);
    try std.testing.expectEqual(@as(usize, 8), y.shape[1]);
}

test "self attention produces valid attention weights" {
    const allocator = std.testing.allocator;

    var attn = try SelfAttention.init(allocator, 4);
    defer attn.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 3, 4 });
    defer x.deinit();

    var y = try attn.forward(&x);
    defer y.deinit();

    // Check cached attention sums to 1 per row
    const attention = &(attn.cached_attention orelse unreachable);
    const seq_len: usize = 3;

    for (0..seq_len) |i| {
        var row_sum: f32 = 0.0;
        for (0..seq_len) |j| {
            row_sum += attention.data[i * seq_len + j];
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), row_sum, 0.001);
    }
}

test "self attention backward shape" {
    const allocator = std.testing.allocator;

    var attn = try SelfAttention.init(allocator, 4);
    defer attn.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 2, 4 });
    defer x.deinit();

    var y = try attn.forward(&x);
    defer y.deinit();

    var grad_out = try Tensor.ones(allocator, &[_]usize{ 2, 4 });
    defer grad_out.deinit();

    var grad_wq = try Tensor.zeros(allocator, &[_]usize{ 4, 4 });
    defer grad_wq.deinit();
    var grad_wk = try Tensor.zeros(allocator, &[_]usize{ 4, 4 });
    defer grad_wk.deinit();
    var grad_wv = try Tensor.zeros(allocator, &[_]usize{ 4, 4 });
    defer grad_wv.deinit();
    var grad_wo = try Tensor.zeros(allocator, &[_]usize{ 4, 4 });
    defer grad_wo.deinit();

    var grad_input = try attn.backward(&grad_out, &grad_wq, &grad_wk, &grad_wv, &grad_wo);
    defer grad_input.deinit();

    try std.testing.expectEqual(@as(usize, 2), grad_input.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), grad_input.shape[1]);
}

test "multihead attention forward shape" {
    const allocator = std.testing.allocator;

    var mha = try MultiHeadAttention.init(allocator, 8, 2);
    defer mha.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 4, 8 });
    defer x.deinit();

    var y = try mha.forward(&x);
    defer y.deinit();

    try std.testing.expectEqual(@as(usize, 4), y.shape[0]);
    try std.testing.expectEqual(@as(usize, 8), y.shape[1]);
}

test "multihead attention invalid head count" {
    const allocator = std.testing.allocator;

    // 8 not divisible by 3
    const result = MultiHeadAttention.init(allocator, 8, 3);
    try std.testing.expectError(error.InvalidHeadCount, result);
}

test "multihead attention attention weights valid" {
    const allocator = std.testing.allocator;

    var mha = try MultiHeadAttention.init(allocator, 4, 2);
    defer mha.deinit();

    var x = try Tensor.ones(allocator, &[_]usize{ 3, 4 });
    defer x.deinit();

    var y = try mha.forward(&x);
    defer y.deinit();

    // Check attention weights sum to 1 per head per row
    const attention = &(mha.cached_attention orelse unreachable);
    const seq_len: usize = 3;

    for (0..mha.num_heads) |h| {
        for (0..seq_len) |i| {
            var row_sum: f32 = 0.0;
            for (0..seq_len) |j| {
                row_sum += attention.data[h * seq_len * seq_len + i * seq_len + j];
            }
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), row_sum, 0.001);
        }
    }
}
