import enum
import numpy as np
import utils


def update_param(dx, learning_rate=1e-2):
    """
    Implementation of standard gradient descent algorithm.
    """
    return learning_rate * dx


def update_param_adagrad(dx, mx, learning_rate=1e-2):
    """
    Implementation of adagrad algorithm.
    """
    return learning_rate * dx / np.sqrt(mx+1e-8)


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


class Layers():
    def __init__(self):
        """
        store: used to store variables and pass information from forward to backward pass. 
        """
        self.store = None


class FullyConnectedLayer(Layers):
    def __init__(self, dim_in, dim_out):
        """
        Implementation of a fully connected layer.

        dim_in: Number of neurons in previous layer.
        dim_out: Number of neurons in current layer.
        w: Weight matrix of the layer.
        b: Bias vector of the layer.
        dw: Gradient of weight matrix.
        db: Gradient of bias vector
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w = np.random.uniform(-1, 1, (dim_in, dim_out)) / max(dim_in, dim_out)
        self.b = np.random.uniform(-1, 1, (dim_out,)) / max(dim_in, dim_out)
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of fully connencted layer.

        x: Input to layer (either of form Nxdim_in or in tensor form after convolution NxCxHxW).
        store: Store input to layer for backward pass.
        """
        self.store = x
        if x.ndim > 2:
            x = np.reshape(x, (x.shape[0], -1))

        bias = np.ones((x.shape[0], 1))

        out = np.hstack((x, bias)) @ np.vstack((self.w, self.b))

        return out

    def backward(self, delta):
        """
        Backward pass of fully connencted layer.

        delta: Error from succeeding layer
        dx: Loss derivitive that that is passed on to layers below
        store: Store input to layer for backward passs
        """
        x = self.store

        if x.ndim > 2:
            x = np.reshape(x, (x.shape[0], -1))

        dx = (self.w @ delta.T).T

        self.dw = x.T @ delta

        self.db = (np.ones((x.shape[0], 1)).T @ delta).flatten()

        # Upades the weights and bias using the computed gradients
        self.w -= update_param(self.dw)
        self.b -= update_param(self.db)

        if self.store.ndim > 2:
            dx = np.reshape(dx, self.store.shape)

        return dx


class ConvolutionalLayer(Layers):
    def __init__(self, filtersize, pad=0, stride=1):
        """
        Implementation of a convolutional layer.

        filtersize = (C_out, C_in, F_H, F_W)
        w: Weight tensor of layer.
        b: Bias vector of layer.
        dw: Gradient of weight tensor.
        db: Gradient of bias vector
        """
        self.filtersize = filtersize
        self.pad = pad
        self.stride = stride
        self.w = np.random.normal(0, 0.1, filtersize)
        self.b = np.random.normal(0, 0.1, (filtersize[0],))
        self.dw = None
        self.db = None

    @staticmethod
    def conv2d(x: np.ndarray, filt: np.ndarray, out_dim, stride):
        out = np.zeros(out_dim)

        if (filt.shape[1] % 2 != 0):
            i_shift = int((filt.shape[1] - 1) / 2)
            j_shift = int((filt.shape[2] - 1) / 2)
        else:
            i_shift = int(filt.shape[1] / 2)
            j_shift = int(filt.shape[2] / 2)

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                i_org = int((i) * stride)
                j_org = int((j) * stride)

                neighbourhood_tensor = \
                    x[:, i_org: i_org+filt.shape[1], j_org: j_org+filt.shape[2]]

                out[i,j] = (neighbourhood_tensor * filt).sum()

        return out

    def forward(self, x):
        """
        Forward pass of convolutional layer.

        x_col: Input tensor reshaped to matrix form.
        store_shape: Save shape of input tensor for backward pass.
        store_col: Save input tensor on matrix from for backward pass.
        """
        N, C, H, W = x.shape
        C_out, C_in, F_H, F_W = self.w.shape

        self.store = x

        Wout = int((W - self.filtersize[3]+2*self.pad)/self.stride+1)
        Hout = int((H - self.filtersize[2]+2*self.pad)/self.stride+1)
        out = np.zeros((N, C_out, Hout, Wout))

        for n, sample in enumerate(x):
            for c, conv_filter in enumerate(self.w):

                out[n, c, :, :] = self.conv2d(sample,
                                              conv_filter,
                                              (Hout, Wout),
                                              self.stride) + self.b[c]

        return out

    def backward(self, delta):
        """
        Backward pass of convolutional layer.

        delta: gradients from layer above
        dx: gradients that are propagated to layer below
        """

        x = self.store

        H_pad = int((self.w.shape[2] - 1) / 2)
        W_pad = int((self.w.shape[3] - 1) / 2)

        delta_pad = np.pad(delta, ((0,0),
                                   (0,0),
                                   (2*H_pad, 2*H_pad),
                                   (2*W_pad, 2*W_pad)))

        w_flipped = np.flip(self.w, axis=(2, 3))

        self.dw = np.zeros_like(self.w)
        dx = np.zeros_like(x)

        #for dx
        for n, sample in enumerate(delta_pad):
            for c, conv_filter in enumerate(np.swapaxes(w_flipped, 0, 1)):

                dx[n, c, :, :] += self.conv2d(sample,
                                              conv_filter,
                                              x.shape[2:],
                                              self.stride)

        # for dw
        for n, sample in enumerate(np.swapaxes(x, 0, 1)):
            for c, conv_filter in enumerate(np.swapaxes(delta, 0, 1)):

                self.dw[c, n, :, :] += self.conv2d(sample,
                                                   conv_filter,
                                                   self.w.shape[2:],
                                                   self.stride)

        # for db
        self.db = delta.sum(axis=(0, 2, 3))
        ######################################################
        ######################################################
        ######################################################

        # Upades the weights and bias using the computed gradients
        self.w -= update_param(self.dw)
        self.b -= update_param(self.db)

        return dx


class MaxPoolingLayer(Layers):
    """
    Implementation of MaxPoolingLayer.
    pool_r, pool_c: integers that denote pooling window size along row and column direction
    stride: integer that denotes with what stride the window is applied
    """

    def __init__(self, pool_r, pool_c, stride):
        self.pool_r = pool_r
        self.pool_c = pool_c
        self.stride = stride

    def forward(self, x):
        """
        Forward pass.
        x: Input tensor of form (NxCxHxW)
        out: Output tensor of form NxCxH_outxW_out
        N: Batch size
        C: Nr of channels
        H, H_out: Input and output heights
        W, W_out: Input and output width
        """
        N, C, H, W = x.shape
        self.store = x

        r_inds = np.arange(0, H-self.pool_r+1, self.stride)
        c_inds = np.arange(0, H-self.pool_c+1, self.stride)

        out = np.zeros((N, C, H//self.pool_r, W//self.pool_c))

        for r in r_inds:
            for c in c_inds:
                out[:, :, r // self.stride, c // self.stride] = \
                    np.max(x[:, :, r: r+self.pool_r, c: c + self.pool_c], axis=(2, 3))

        return out

    def backward(self, delta):
        """
        Backward pass.
        delta: loss derivative from above (of size NxCxH_outxW_out)
        dX: gradient of loss wrt. input (of size NxCxHxW)
        """

        x = self.store
        N, C, H, W = x.shape
        r_inds = np.arange(0, H-self.pool_r+1, self.stride)
        c_inds = np.arange(0, H-self.pool_c+1, self.stride)

        dx = np.zeros_like(x)

        for i, sample in enumerate(x):
            for j, filt in enumerate(sample):
                for k, r in enumerate(r_inds):
                    for l, c in enumerate(c_inds):
                        local_max_ind = np.unravel_index(np.argmax(
                            filt[r: r+self.pool_r, c: c + self.pool_c]), (self.pool_r, self.pool_c))

                        dx[i, j, r+local_max_ind[0], c+local_max_ind[1]] = delta[i,j,k,l]

        return dx


class LSTMLayer(Layers):
    """
    Implementation of a LSTM layer.

    dim_in: Integer indicating input dimension
    dim_hid: Integer indicating hidden dimension
    wx: Weight tensor for input to hidden mapping (dim_in, 4*dim_hid)
    wh: Weight tensor for hidden to hidden mapping (dim_hid, 4*dim_hid)
    b: Bias vector of layer (4*dim_hid)
    """

    def __init__(self, dim_in, dim_hid):
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.wx = np.random.normal(0, 0.1, (dim_in, 4*dim_hid))
        self.wh = np.random.normal(0, 0.1, (dim_hid, 4*dim_hid))
        self.b = np.random.normal(0, 0.1, (4*dim_hid,))

    def forward_step(self, x, h, c):
        """
        Implementation of a single forward step (one timestep)
        x: Input to layer (Nxdim_in) where N=#samples in batch and dim_in=feature dimension
        h: Hidden state from previous time step (Nxdim_hid) where dim_hid=#hidden units
        c: Cell state from previous time step (Nxdim_hid) where dim_hid=#hidden units
        next_h: Updated hidden state(Nxdim_hid)
        next_c: Updated cell state(Nxdim_hid)
        cache: A tuple where you can store anything that might be useful for the backward pass
        """

	    ######################################################
        ######## REPLACE NEXT PART WITH YOUR SOLUTION ########
        ######################################################
        next_h = np.random.random_sample(h.shape)
        next_c = np.random.random_sample(c.shape)
        cache = (next_h, next_c)
        ######################################################
        ######################################################
        ######################################################

        return next_h, next_c, cache

    def backward_step(self, delta_h, delta_c, store):
        """
        Implementation of a single backward step (one timestep)
        delta_h: Upstream gradients from hidden state
        delta_h: Upstream gradients from cell state
        store:
          hn: Updated hidden state from forward pass (Nxdim_hid) where dim_hid=#hidden units
          x: Input to layer (Nxdim_in) where N=#samples in batch and dim_in=feature dimension
          h: Hidden state from previous time step (Nxdim_hid) where dim_hid=#hidden units
          cn: Updated cell state from forward pass (Nxdim_hid) where dim_hid=#hidden units
          c: Cell state from previous time step (Nxdim_hid) where dim_hid=#hidden units
          cache: Whatever was added to the cache in forward pass
        dx: Gradient of loss wrt. input
        dh: Gradient of loss wrt. previous hidden state
        dc: Gradient of loss wrt. previous cell state
        dwh: Gradient of loss wrt. weight tensor for hidden to hidden mapping
        dwx: Gradient of loss wrt. weight tensor for input to hidden mapping
        db: Gradient of loss wrt. bias vector
        """
        hn, x, h, cn, c, cache = store

        ######################################################
        ######## REPLACE NEXT PART WITH YOUR SOLUTION ########
        ######################################################
        dx = np.random.random_sample(x.shape)
        dh = np.random.random_sample(h.shape)
        dc = np.random.random_sample(c.shape)
        dwh = np.random.random_sample(self.wh.shape)
        dwx = np.random.random_sample(self.wx.shape)
        db = np.random.random_sample(self.b.shape)
        ######################################################
        ######################################################
        ######################################################

        return dx, dh, dc, dwh, dwx, db


class WordEmbeddingLayer(Layers):
    """
    Implementation of WordEmbeddingLayer.
    """

    def __init__(self, vocab_dim, embedding_dim):
        self.w = np.random.normal(0, 0.1, (vocab_dim, embedding_dim))
        self.dw = None

    def forward(self, x):
        """
        Forward pass.
        Look-up embedding for x of form (NxTx1) where each element is an integer indicating the word id.
        N: Number of words in batch. 
        T: Number of timesteps.
        Output: (NxTxE) where E is embedding dimensionality.
        """
        self.store = x
        return self.w[x,:]

    def backward(self, delta):
        """
        Backward pass. Update embedding matrix.
        Delta: Loss derivative from above
        """
        x = self.store
        self.dw = np.zeros(self.w.shape)
        np.add.at(self.dw, x, delta)
        self.w -= update_param(self.dw)
        return 0


"""
Activation functions
"""

class SoftmaxLossLayer(Layers):
    """
    Implementation of SoftmaxLayer forward pass with cross-entropy loss.
    """
    def forward(self, x, y):
        ex = np.exp(x-np.max(x, axis=1, keepdims=True))
        y_hat = ex/np.sum(ex, axis=1, keepdims=True)
        m = y.shape[0]
        log_likehood = -np.log(y_hat[range(m), y.astype(int)])
        loss = np.sum(log_likehood) / m

        d_out = y_hat
        d_out[range(m), y.astype(int)] -= 1
        d_out /= m

        return loss, d_out

class SoftmaxLayer(Layers):
    """
    Implementation of SoftmaxLayer forward pass.
    """

    def forward(self, x):
        ex = np.exp(x-np.max(x, axis=1, keepdims=True))
        y_hat = ex/np.sum(ex, axis=1, keepdims=True)
        return y_hat


class ReluLayer(Layers):
    """
    Implementation of relu activation function.
    """
    def forward(self, x):
        """
        x: Input to layer. Any dimension.
        """
        ######################################################
        ######## REPLACE NEXT PART WITH YOUR SOLUTION ########
        ######################################################
        self.store = x
        out = np.where(x>0, x, 0)
        ######################################################
        ######################################################
        ######################################################
        return out

    def backward(self, delta):
        """
        delta: Loss derivative from above. Any dimension.
        """
        ######################################################
        ######## REPLACE NEXT PART WITH YOUR SOLUTION ########
        ######################################################
        x = self.store
        dx = np.where(x>0, delta, 0) 
        ######################################################
        ######################################################
        ######################################################
        return dx



