package gpt2

// tensor is a wrapper around a slice of float32 values and a list of dimensions
type tensor struct {
	data []float32
	dims []int
}

// newTensor creates a new tensor with the given data and dimensions.
func newTensor(data []float32, dims ...int) (tensor, int) {
	s := 1
	for _, d := range dims {
		s *= d
	}
	if s > len(data) {
		panic("dimensions larger than supplied data")
	}
	ss := min(s, len(data))
	return tensor{
		data: data[:ss],
		dims: dims,
	}, ss
}

// sampleMult returns the index of the first element of probabilities that is greater than or equal to coin.
func sampleMult(probabilities []float32, coin float32) int {
	var cdf float32
	for i, prob := range probabilities {
		cdf += prob
		if coin < cdf {
			return i
		}
	}
	return len(probabilities) - 1
}

// BatchActivationTensors is a struct that contains all the activation tensors for a single batch of input data.
type BatchActivationTensors struct {
	Memory             []float32
	Encoded            tensor // (B, T, C) - Initial encoded input representations of the batch (Batch size, Sequence length, Embedding dimension).
	Layer1Act          tensor // (L, B, T, C) - Activations after Layer Normalization 1 of the batch.
	LayerNorm1Mean     tensor // (L, B, T) - Mean values for Layer Normalization 1 of the batch.
	LayerNorm1Rstd     tensor // (L, B, T) - Reciprocal of standard deviation for Layer Normalization 1 of the batch.
	QueryKeyVal        tensor // (L, B, T, 3*C) - Combined Query, Key, Value representations for the attention mechanism.  (Layers, Batch size, Sequence length, 3 * Embedding dimension).
	AttentionInter     tensor // (L, B, T, C) - Intermediate attention-like result of the batch.
	PreAttention       tensor // (L, B, NH, T, T) - Pre-attention scores of the batch.
	Attention          tensor // (L, B, NH, T, T) - Normalized attention weights (Number of layers, Batch size, Number of Attention Heads, Sequence length, Sequence length).
	AttentionProj      tensor // (L, B, T, C) - Projected attention outputs of the batch.
	Residual2          tensor // (L, B, T, C) - Residual connection after attention of the batch.
	LayerNorm2Act      tensor // (L, B, T, C) - Activations after Layer Normalization 2 of the batch.
	LayerNorm2Mean     tensor // (L, B, T) - Mean values for Layer Normalization 2 of the batch.
	LayerNorm2Rstd     tensor // (L, B, T) - Reciprocal of standard deviation for Layer Normalization 2 of the batch.
	FeedForward        tensor // (L, B, T, 4*C) - Intermediate Feed-Forward Network activations of the batch.
	FeedForwardGelu    tensor // (L, B, T, 4*C) - FeedForward activations after applying GELU (non-linearity) of the batch.
	FeedForwardProj    tensor // (L, B, T, C) - Projected output of the Feed-Forward Network of the batch.
	Residual3          tensor // (L, B, T, C) - Residual connection after Feed-Forward Network of the batch.
	LayerNormFinal     tensor // (B, T, C) - Final activations after Layer Normalization of the batch.
	LayerNormFinalMean tensor // (B, T) - Mean values for final Layer Normalization of the batch.
	LayerNormFinalStd  tensor // (B, T) - Reciprocal of standard deviation for final Layer Normalization of the batch.
	Logits             tensor // (B, T, V) - Raw output scores (before softmax) of the batch.
	Probabilities      tensor // (B, T, V) - Softmax probabilities over the vocabulary of the batch.
	Losses             tensor // (B, T) - Loss values per token in the batch.
}

// Init initialises the ActivationTensors with specific sizes for each tensor based on the model architecture.
func (tensor *BatchActivationTensors) Init(B, C, T, L, NH, V int) {
	tensor.Memory = make([]float32,
		B*T*C+
			L*B*T*C+
			L*B*T+
			L*B*T+
			L*B*T*C*3+
			L*B*T*C+
			L*B*NH*T*T+
			L*B*NH*T*T+
			L*B*T*C+
			L*B*T*C+
			L*B*T*C+
			L*B*T+
			L*B*T+
			L*B*T*C*4+
			L*B*T*C*4+
			L*B*T*C+
			L*B*T*C+
			B*T*C+
			B*T+
			B*T+
			B*T*V+
			B*T*V+
			B*T)
	var ptr int
	memPtr := tensor.Memory
	tensor.Encoded, ptr = newTensor(memPtr, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Layer1Act, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm1Mean, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm1Rstd, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.QueryKeyVal, ptr = newTensor(memPtr, L, B, T, C*3)
	memPtr = memPtr[ptr:]
	tensor.AttentionInter, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.PreAttention, ptr = newTensor(memPtr, L, B, NH, T, T)
	memPtr = memPtr[ptr:]
	tensor.Attention, ptr = newTensor(memPtr, L, B, NH, T, T)
	memPtr = memPtr[ptr:]
	tensor.AttentionProj, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Residual2, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm2Act, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm2Mean, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm2Rstd, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.FeedForward, ptr = newTensor(memPtr, L, B, T, C*4)
	memPtr = memPtr[ptr:]
	tensor.FeedForwardGelu, ptr = newTensor(memPtr, L, B, T, C*4)
	memPtr = memPtr[ptr:]
	tensor.FeedForwardProj, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Residual3, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNormFinal, ptr = newTensor(memPtr, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNormFinalMean, ptr = newTensor(memPtr, B, T)
	memPtr = memPtr[ptr:]
	tensor.LayerNormFinalStd, ptr = newTensor(memPtr, B, T)
	memPtr = memPtr[ptr:]
	tensor.Logits, ptr = newTensor(memPtr, B, T, V)
	memPtr = memPtr[ptr:]
	tensor.Probabilities, ptr = newTensor(memPtr, B, T, V)
	memPtr = memPtr[ptr:]
	tensor.Losses, ptr = newTensor(memPtr, B, T)
	memPtr = memPtr[ptr:]
	if len(memPtr) != 0 {
		panic("something went real bad here")
	}
}

// ParameterTensors are the parameters of the model.
type ParameterTensors struct {
	Memory        []float32
	WordTokEmbed  tensor // (V, C) - Word/Token Embedding weights (Vocabulary size, Embedding dimension).
	WordPosEmbed  tensor // (maxT, C) - Positional Embedding weights (Maximum Sequence length, Embedding dimension).
	LayerNorm1W   tensor // (L, C) - Weights for Layer Normalization 1 (Number of layers, Embedding dimension).
	LayerNorm1B   tensor // (L, C) - Model Biases for Layer Normalization 1.
	QueryKeyValW  tensor // (L, 3*C, C) - Attention QKV weights (Layers, 3 * Embedding dimension, Embedding dimension).
	QueryKeyValB  tensor // (L, 3*C) - Model Attention QKV biases.
	AttProjW      tensor // (L, C, C) - Attention projection weights (Layers, Embedding dimension, Embedding dimension).
	AttProjB      tensor // (L, C) - Model Attention projection biases.
	Layer2NormW   tensor // (L, C) - Model weights for Layer Normalization 2.
	Layer2NormB   tensor // (L, C) - Model Biases for Layer Normalization 2.
	FeedFwdW      tensor // (L, 4*C, C) - Feed-forward layer weights (Layers, 4 * Embedding Dimension, Embedding Dimension).
	FeedFwdB      tensor // (L, 4*C) - Model Feed-forward layer biases.
	FeedFwdProjW  tensor // (L, C, 4*C) - Model Feed-forward projection weights.
	FeedFwdProjB  tensor // (L, C)- Model Feed-forward projection biases.
	LayerFinNormW tensor // (C) - Model Final layer normalization weights.
	LayerFinNormB tensor // (C) - Model Final layer normalization biases.
}

// Init initialises the ParameterTensors with specific sizes for each tensor based on the model architecture.
func (tensor *ParameterTensors) Init(V, C, maxSeqLen, L int) {
	tensor.Memory = make([]float32,
		V*C+ // WordTokEmbed
			maxSeqLen*C+ // WordPosEmbed
			L*C+ // LayerNorm1W
			L*C+ // LayerNorm1B
			L*3*C*C+ // QueryKeyValW
			L*3*C+ // QueryKeyValB
			L*C*C+ // AttProjW
			L*C+ // AttProjB
			L*C+ // Layer2NormW
			L*C+ // Layer2NormB
			L*4*C*C+ // FeedFwdW
			L*4*C+ // FeedFwdB
			L*C*4*C+ // FeedFwdProjW
			L*C+ // FeedFwdProjB
			C+ // LayerFinNormW
			C, // LayerFinNormB
	)
	var ptr int
	memPtr := tensor.Memory
	tensor.WordTokEmbed, ptr = newTensor(memPtr, V, C)
	memPtr = memPtr[ptr:]
	tensor.WordPosEmbed, ptr = newTensor(memPtr, maxSeqLen, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm1W, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm1B, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.QueryKeyValW, ptr = newTensor(memPtr, L, 3*C, C)
	memPtr = memPtr[ptr:]
	tensor.QueryKeyValB, ptr = newTensor(memPtr, L, 3*C)
	memPtr = memPtr[ptr:]
	tensor.AttProjW, ptr = newTensor(memPtr, L, C, C)
	memPtr = memPtr[ptr:]
	tensor.AttProjB, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.Layer2NormW, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.Layer2NormB, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdW, ptr = newTensor(memPtr, L, 4*C, C)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdB, ptr = newTensor(memPtr, L, 4*C)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdProjW, ptr = newTensor(memPtr, L, C, 4*C)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdProjB, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.LayerFinNormW, ptr = newTensor(memPtr, C)
	memPtr = memPtr[ptr:]
	tensor.LayerFinNormB, ptr = newTensor(memPtr, C)
	memPtr = memPtr[ptr:]
	if len(memPtr) != 0 {
		panic("something went real bad here")
	}
}

// Len returns the length of the memory slice.
func (tensor *ParameterTensors) Len() int {
	return len(tensor.Memory)
}
