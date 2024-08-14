package gpt2

import (
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"
	"time"

	"github.com/conneroisu/gpt2/pkg/data"
	"github.com/conneroisu/gpt2/pkg/torch"
)

const (
	GPT2_EOT int32 = 50256

	genMaxLength, valNumBatches = 64, 10
)

// GPT2Config is a configuration struct for the GPT-2 model.
type GPT2Config struct {
	// MaxSeqLen is the maximum sequence length for the model.
	MaxSeqLen int
	// VocabSize is the size of the vocabulary.
	VocabSize int
	// NumLayers is the number of layers in the model.
	NumLayers int
	// NumHeads is the number of attention heads in each layer.
	NumHeads int
	// Channels is the number of channels in each layer.
	Channels int
	// EOT is the end-of-text token ID.
	EOT int32
}

// Optimizer is an interface for optimizers.
type Optimizer interface{}

// AdamW is an implementation of the AdamW optimizer.
type AdamW struct {
	// FirstMomentEstimates is a array of first moment estimates.
	FirstMomentEstimates []float32
	// SecondMomentEstimates is a array of second moment estimates.
	SecondMomentEstimates []float32
	// Activations is a tensor of activations.
	Activations BatchActivationTensors
}

// GPT is the interface for general pretrained transformer models.
type GPT interface {
	// Forward performs a forward pass on the model.
	Forward(input, targets []int32, batchSize, t int)
	// Backward performs a backward pass on the model.
	Backward() error
	// Update performs an update on the model.
	Update(learningRate, beta1, beta2, epsilon, weightDecay float32, t int)
	// Inference performs an inference on the model.
	Inference(input string, batchSize, t int)
}

// GPT2 is a GPT-2 model.
type GPT2 struct {
	// Tokenizer is the tokenizer used to tokenize the input text.
	Tokenizer GPT2Tokenizer
	// Config is the configuration of the model.
	Config GPT2Config
	// Params is the parameters of the model.
	Params ParameterTensors
	// Gradients is the gradients of the model to be applied to the parameters.
	Gradients ParameterTensors
	// Optimizer is the optimizer used to update the parameters.
	Optimizer AdamW
	// GradientActivations is the activations of the model.
	GradientActivations BatchActivationTensors
	// BatchSize is the batch size of the model.
	BatchSize int
	// SequenceLength is the sequence length of the model.
	SequenceLength int
	// Inputs is the input of the model.
	Inputs []int32
	// Targets are the target tokens of the model.
	Targets []int32
	// MeanLoss is the mean loss of the model.
	MeanLoss float32
}

// NewGPT2 creates a new GPT-2 model from a reader.
func NewGPT2(r io.Reader) (*GPT2, error) {
	header := make([]int32, 256)
	err := binary.Read(r, binary.LittleEndian, &header)
	if err != nil {
		return nil, err
	}
	if header[0] != 20240326 || header[1] != 1 {
		return nil, fmt.Errorf("Invalid header")
	}
	model := &GPT2{
		Config: GPT2Config{
			MaxSeqLen: int(header[2]),
			VocabSize: int(header[3]),
			NumLayers: int(header[4]),
			NumHeads:  int(header[5]),
			Channels:  int(header[6]),
			EOT:       GPT2_EOT,
		},
	}
	model.Params.Init(
		model.Config.VocabSize,
		model.Config.Channels,
		model.Config.MaxSeqLen,
		model.Config.NumLayers,
	)
	err = binary.Read(r, binary.LittleEndian, model.Params.Memory)
	if err != nil {
		return nil, fmt.Errorf("error reading model: %v", err)
	}
	return model, nil
}

// LoadModel loads a GPT-2 model from a checkpoint file.
//
// The checkpoint file is a binary file that contains the model parameters and gradients.
//
// The binPath file is the tokenizer file.
func LoadModel(ckptPath, binPath string) (*GPT2, error) {
	if ckptPath == "" {
		return nil, fmt.Errorf("checkpoint file path is required")
	}
	if binPath == "" {
		return nil, fmt.Errorf("tokenizer file path is required")
	}
	f, err := os.Open(ckptPath)
	if err != nil {
		return nil, fmt.Errorf("Error opening model file: %v", err)
	}
	defer f.Close()
	model, err := NewGPT2(f)
	if err != nil {
		return nil, err
	}
	tok, err := NewTokenizer(binPath)
	if err != nil {
		return nil, err
	}
	model.Tokenizer = tok
	return model, nil
}

// Inference performs an inference on the model.
func (model *GPT2) Inference(input string, B, T int) (string, error) {
	start := time.Now()
	defer func() {
		fmt.Printf("inference time took: %v\n", time.Since(start))
	}()
	tokens, err := model.Tokenizer.Encode(input)
	if err != nil {
		return "", err
	}
	if len(tokens) < T {
		for i := len(tokens); i <= T; i++ {
			tokens = append(tokens, model.Config.EOT)
		}
	}
	fmt.Printf("input is %d tokens long\n", len(tokens))
	model.Forward(tokens, tokens[1:], B, T)
	genTokens := make([]int32, B*T)
	for i := 0; i < B*T; i++ {
		genTokens[i] = model.Config.EOT
	}
	for t := 1; t < B*T; t++ {
		fmt.Printf("generating token: %d\n", t)
		// Recompute all activations between 0 and t
		model.Forward(genTokens, nil, B, t)
		probabilities := model.Optimizer.Activations.Probabilities.data[(t-1)*model.Config.VocabSize:]
		coin := rand.Float32()
		nextToken2 := sampleMult(probabilities, coin)
		genTokens[t] = rune(nextToken2)
	}
	if model.Tokenizer.init {
		return model.Tokenizer.Decode(genTokens)
	}
	return "", fmt.Errorf("tokenizer not initialised")
}

// Update performs an update on the model.
func (model *GPT2) Update(learningRate, beta1, beta2, eps, weightDecay float32, t int) {
	if model.Optimizer.FirstMomentEstimates == nil {
		model.Optimizer.FirstMomentEstimates = make([]float32, model.Params.Len())
		model.Optimizer.SecondMomentEstimates = make([]float32, model.Params.Len())
	}
	// Parameter updates
	for i := 0; i < model.Params.Len(); i++ {
		parameter := model.Params.Memory[i]
		gradient := model.Gradients.Memory[i]
		// update the momentum (m is the updated first moment estimate)
		m := beta1*model.Optimizer.FirstMomentEstimates[i] + (1.0-beta1)*gradient
		// RMSprop update (v is the updated second moment estimate)
		v := beta2*model.Optimizer.SecondMomentEstimates[i] + (1.0-beta2)*gradient*gradient
		// correct the bias
		mHat := m / (1.0 - torch.Pow(beta1, float32(t)))
		vHat := v / (1.0 - torch.Pow(beta2, float32(t)))
		// update the parameters
		model.Optimizer.FirstMomentEstimates[i] = m
		model.Optimizer.SecondMomentEstimates[i] = v
		model.Params.Memory[i] -= learningRate * (mHat/(torch.Sqrt(vHat)+eps) + weightDecay*parameter)
	}
}

// Backward performs a backward pass on the model.
func (model *GPT2) Backward() error {
	// assert we forwarded previously
	if model.MeanLoss == -1.0 {
		return fmt.Errorf("error: must forward before backward")
	}

	if len(model.Gradients.Memory) == 0 {
		model.Gradients.Init(model.Config.VocabSize, model.Config.Channels, model.Config.MaxSeqLen, model.Config.NumLayers)
		model.GradientActivations.Init(model.BatchSize, model.Config.Channels, model.SequenceLength, model.Config.NumLayers, model.Config.NumHeads, model.Config.VocabSize)
		model.ZeroGradient()
	}
	// get mean loss by filling gradient losses with 1.0f/(B*T)
	dlossMean := 1.0 / float32(model.BatchSize*model.SequenceLength)
	for i := range model.GradientActivations.Losses.data {
		model.GradientActivations.Losses.data[i] = dlossMean
	}
	torch.CrossentropySoftmaxBackward(
		model.GradientActivations.Logits.data,
		model.GradientActivations.Losses.data,
		model.Optimizer.Activations.Probabilities.data,
		model.Targets,
		model.BatchSize,
		model.SequenceLength,
		model.Config.VocabSize,
	)
	torch.MatmulBackward(
		model.GradientActivations.LayerNormFinal.data,
		model.Gradients.WordTokEmbed.data,
		nil,
		model.GradientActivations.Logits.data,
		model.Optimizer.Activations.LayerNormFinal.data,
		model.Params.WordTokEmbed.data,
		model.BatchSize,
		model.SequenceLength,
		model.Config.Channels,
		model.Config.VocabSize,
	)
	// last layer's residual (the gradient is written to the last layer's residual)
	residual := model.
		Optimizer.
		Activations.
		Residual3.
		data[(model.Config.NumLayers-1)*model.BatchSize*model.SequenceLength*model.Config.Channels:]
	// write the gradient to the last layer's residual
	dresidual := model.
		GradientActivations.
		Residual3.
		data[(model.Config.NumLayers-1)*model.BatchSize*model.SequenceLength*model.Config.Channels:]
	torch.LayernormBackward(
		dresidual,
		model.Gradients.LayerFinNormW.data,
		model.Gradients.LayerFinNormB.data,
		model.GradientActivations.LayerNormFinal.data,
		residual,
		model.Params.LayerFinNormW.data,
		model.Optimizer.Activations.LayerNormFinalMean.data,
		model.Optimizer.Activations.LayerNormFinalStd.data,
		model.BatchSize,
		model.SequenceLength,
		model.Config.Channels,
	)
	for layer := model.Config.NumLayers - 1; layer >= 0; layer-- {
		if layer == 0 {
			residual = model.
				Optimizer.
				Activations.
				Encoded.
				data
			dresidual = model.
				GradientActivations.
				Encoded.
				data
		} else {
			residual = model.
				Optimizer.
				Activations.
				Residual3.
				data[(layer-1)*model.
				BatchSize*model.
				SequenceLength*model.
				Config.
				Channels:]
			dresidual = model.
				GradientActivations.
				Residual3.
				data[(layer-1)*model.
				BatchSize*model.
				SequenceLength*model.
				Config.
				Channels:]
		}
		l_ln1w := model.
			Params.
			LayerNorm1W.
			data[layer*model.
			Config.
			Channels:]
		l_qkvw := model.
			Params.
			QueryKeyValW.
			data[layer*3*model.
			Config.
			Channels*model.
			Config.
			Channels:]
		l_attprojw := model.
			Params.
			AttProjW.
			data[layer*model.
			Config.
			Channels*model.
			Config.
			Channels:]
		l_ln2w := model.
			Params.
			Layer2NormW.
			data[layer*model.
			Config.
			Channels:]
		l_fcw := model.
			Params.
			FeedFwdW.
			data[layer*4*model.
			Config.
			Channels*model.
			Config.
			Channels:]
		l_fcprojw := model.
			Params.
			FeedFwdProjW.
			data[layer*model.
			Config.
			Channels*4*model.
			Config.
			Channels:]
		// Gradients of weights (dl_* is the gradient of the loss with respect to the parameter)
		dl_ln1w := model.
			Gradients.
			LayerNorm1W.
			data[layer*model.
			Config.
			Channels:]
		dl_ln1b := model.
			Gradients.
			LayerNorm1B.
			data[layer*model.
			Config.
			Channels:]
		dl_qkvw := model.
			Gradients.
			QueryKeyValW.
			data[layer*3*model.
			Config.
			Channels*model.
			Config.
			Channels:]
		dl_qkvb := model.
			Gradients.
			QueryKeyValB.
			data[layer*3*model.
			Config.
			Channels:]
		dl_attprojw := model.
			Gradients.
			AttProjW.
			data[layer*model.
			Config.
			Channels*model.
			Config.
			Channels:]
		dl_attprojb := model.
			Gradients.
			AttProjB.
			data[layer*model.
			Config.
			Channels:]
		dl_ln2w := model.
			Gradients.
			Layer2NormW.
			data[layer*model.
			Config.
			Channels:]
		dl_ln2b := model.
			Gradients.
			Layer2NormB.
			data[layer*model.
			Config.
			Channels:]
		dl_fcw := model.
			Gradients.
			FeedFwdW.
			data[layer*4*model.
			Config.
			Channels*model.
			Config.
			Channels:]
		dl_fcb := model.
			Gradients.
			FeedFwdB.
			data[layer*4*model.
			Config.
			Channels:]
		dl_fcprojw := model.
			Gradients.
			FeedFwdProjW.
			data[layer*model.
			Config.
			Channels*4*model.
			Config.
			Channels:]
		dl_fcprojb := model.
			Gradients.
			FeedFwdProjB.
			data[layer*model.
			Config.
			Channels:]
		// Activations (l_* is the output of the layer)
		l_ln1 := model.
			Optimizer.
			Activations.
			Layer1Act.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		l_ln1_mean := model.
			Optimizer.
			Activations.
			LayerNorm1Mean.
			data[layer*model.
			BatchSize*model.
			SequenceLength:]
		l_ln1_rstd := model.
			Optimizer.
			Activations.
			LayerNorm1Rstd.
			data[layer*model.
			BatchSize*model.
			SequenceLength:]
		l_qkv := model.
			Optimizer.
			Activations.
			QueryKeyVal.
			data[layer*model.
			BatchSize*model.
			SequenceLength*3*model.
			Config.
			Channels:]
		l_atty := model.
			Optimizer.
			Activations.
			AttentionInter.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		l_att := model.
			Optimizer.
			Activations.
			Attention.
			data[layer*model.
			BatchSize*model.
			Config.
			NumHeads*model.
			SequenceLength*model.
			SequenceLength:]
		l_residual2 := model.
			Optimizer.
			Activations.
			Residual2.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		l_ln2 := model.
			Optimizer.
			Activations.
			LayerNorm2Act.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		l_ln2_mean := model.
			Optimizer.
			Activations.
			LayerNorm2Mean.
			data[layer*model.
			BatchSize*model.
			SequenceLength:]
		l_ln2_rstd := model.
			Optimizer.
			Activations.
			LayerNorm2Rstd.
			data[layer*model.
			BatchSize*model.
			SequenceLength:]
		l_fch := model.
			Optimizer.
			Activations.
			FeedForward.
			data[layer*model.
			BatchSize*model.
			SequenceLength*4*model.
			Config.
			Channels:]
		l_fch_gelu := model.
			Optimizer.
			Activations.
			FeedForwardGelu.
			data[layer*model.
			BatchSize*model.
			SequenceLength*4*model.
			Config.
			Channels:]
		dl_ln1 := model.
			GradientActivations.
			Layer1Act.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		dl_qkv := model.
			GradientActivations.
			QueryKeyVal.
			data[layer*model.
			BatchSize*model.
			SequenceLength*3*model.
			Config.
			Channels:]
		dl_atty := model.
			GradientActivations.
			AttentionInter.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		dl_preatt := model.
			GradientActivations.
			PreAttention.
			data[layer*model.
			BatchSize*model.
			Config.
			NumHeads*model.
			SequenceLength*model.
			SequenceLength:]
		dl_att := model.
			GradientActivations.
			Attention.
			data[layer*model.
			BatchSize*model.
			Config.
			NumHeads*model.
			SequenceLength*model.
			SequenceLength:]
		dl_attproj := model.
			GradientActivations.
			AttentionProj.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		dl_residual2 := model.
			GradientActivations.
			Residual2.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		dl_ln2 := model.
			GradientActivations.
			LayerNorm2Act.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		dl_fch := model.
			GradientActivations.
			FeedForward.
			data[layer*model.
			BatchSize*model.
			SequenceLength*4*model.
			Config.
			Channels:]
		dl_fch_gelu := model.
			GradientActivations.
			FeedForwardGelu.
			data[layer*model.
			BatchSize*model.
			SequenceLength*4*model.
			Config.
			Channels:]
		dl_fcproj := model.
			GradientActivations.
			FeedForwardProj.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		dl_residual3 := model.
			GradientActivations.
			Residual3.
			data[layer*model.
			BatchSize*model.
			SequenceLength*model.
			Config.
			Channels:]
		torch.ResidualBackward(
			dl_residual2,
			dl_fcproj,
			dl_residual3,
			model.BatchSize*model.SequenceLength*model.Config.Channels,
		)
		torch.MatmulBackward(
			dl_fch_gelu,
			dl_fcprojw,
			dl_fcprojb,
			dl_fcproj,
			l_fch_gelu,
			l_fcprojw,
			model.BatchSize,
			model.SequenceLength,
			4*model.Config.Channels,
			model.Config.Channels,
		)
		torch.GeluBackward(
			dl_fch,
			l_fch,
			dl_fch_gelu,
			model.BatchSize*model.SequenceLength*4*model.Config.Channels,
		)
		torch.MatmulBackward(
			dl_ln2,
			dl_fcw,
			dl_fcb,
			dl_fch,
			l_ln2,
			l_fcw,
			model.BatchSize,
			model.SequenceLength,
			model.Config.Channels,
			4*model.Config.Channels,
		)
		torch.LayernormBackward(
			dl_residual2,
			dl_ln2w,
			dl_ln2b,
			dl_ln2,
			l_residual2,
			l_ln2w,
			l_ln2_mean,
			l_ln2_rstd,
			model.BatchSize,
			model.SequenceLength,
			model.Config.Channels,
		)
		torch.ResidualBackward(
			dresidual,
			dl_attproj,
			dl_residual2,
			model.BatchSize*model.SequenceLength*model.Config.Channels,
		)
		torch.MatmulBackward(
			dl_atty,
			dl_attprojw,
			dl_attprojb,
			dl_attproj,
			l_atty,
			l_attprojw,
			model.BatchSize,
			model.SequenceLength,
			model.Config.Channels,
			model.Config.Channels,
		)
		torch.AttentionBackward(
			dl_qkv,
			dl_preatt,
			dl_att,
			dl_atty,
			l_qkv,
			l_att,
			model.BatchSize,
			model.SequenceLength,
			model.Config.Channels,
			model.Config.NumHeads,
		)
		torch.MatmulBackward(
			dl_ln1,
			dl_qkvw,
			dl_qkvb,
			dl_qkv,
			l_ln1,
			l_qkvw,
			model.BatchSize,
			model.SequenceLength,
			model.Config.Channels,
			3*model.Config.Channels,
		)
		torch.LayernormBackward(
			dresidual,
			dl_ln1w,
			dl_ln1b,
			dl_ln1,
			residual,
			l_ln1w,
			l_ln1_mean,
			l_ln1_rstd,
			model.BatchSize,
			model.SequenceLength,
			model.Config.Channels,
		)
	}
	// Here we want to apply our gradients to our encoded data.
	torch.EncoderBackward(
		model.Gradients.WordTokEmbed.data,
		model.Gradients.WordPosEmbed.data,
		model.GradientActivations.Encoded.data,
		model.Inputs,
		model.BatchSize,
		model.SequenceLength,
		model.Config.Channels,
	)
	return nil
}

// Forward performs a forward pass on the model.
func (model *GPT2) Forward(input, target []int32, B, T int) {
	if model.Optimizer.Activations.Memory == nil {
		model.BatchSize, model.SequenceLength = B, T
		model.Optimizer.Activations.Init(
			B,
			model.Config.Channels,
			T,
			model.Config.NumLayers,
			model.Config.NumHeads,
			model.Config.VocabSize,
		)
		model.Inputs = make([]int32, B*T)
		model.Targets = make([]int32, B*T)
	}
	copy(model.Inputs, input)
	copy(model.Targets, target)
	// Encode the word token embeddings with the positional embeddings
	// so that those vectors have spacial information and aren't just purely made up of the
	// token embeddings.
	//
	// Input is a slice of ids/tokens that correspond to the vectors in word token embeding and their index is the "position"
	torch.EncoderForward(
		model.Optimizer.Activations.Encoded.data,
		input,
		model.Params.WordTokEmbed.data,
		model.Params.WordPosEmbed.data,
		B,
		T,
		model.Config.Channels,
	)
	var residual []float32
	// Apply the residual connections from each layer.
	for layer := 0; layer < model.Config.NumLayers; layer++ {
		// residual is either the connection between the last layers output, or the initial token/pos embedding.
		if layer == 0 {
			residual = model.Optimizer.Activations.Encoded.data
		} else {
			residual = model.Optimizer.Activations.Residual3.data[(layer-1)*B*T*model.Config.Channels:]
		}
		// Parameters (l_*w and l_*b are the weight and bias of the layer)
		l_ln1w := model.
			Params.
			LayerNorm1W.
			data[layer*model.
			Config.
			Channels:]
		l_ln1b := model.
			Params.
			LayerNorm1B.
			data[layer*model.
			Config.
			Channels:]
		l_qkvw := model.
			Params.
			QueryKeyValW.
			data[layer*3*model.
			Config.
			Channels*model.
			Config.
			Channels:]
		l_qkvb := model.
			Params.
			QueryKeyValB.
			data[layer*3*model.
			Config.
			Channels:]
		l_attprojw := model.
			Params.
			AttProjW.
			data[layer*model.
			Config.
			Channels*model.
			Config.
			Channels:]
		l_attprojb := model.
			Params.
			AttProjB.
			data[layer*model.
			Config.
			Channels:]
		l_ln2w := model.
			Params.
			Layer2NormW.
			data[layer*model.
			Config.
			Channels:]
		l_ln2b := model.
			Params.
			Layer2NormB.
			data[layer*model.
			Config.
			Channels:]
		l_fcw := model.
			Params.
			FeedFwdW.
			data[layer*4*model.
			Config.
			Channels*model.
			Config.
			Channels:]
		l_fcb := model.
			Params.
			FeedFwdB.
			data[layer*4*model.
			Config.
			Channels:]
		l_fcprojw := model.
			Params.
			FeedFwdProjW.
			data[layer*model.
			Config.
			Channels*4*model.
			Config.
			Channels:]
		l_fcprojb := model.
			Params.
			FeedFwdProjB.
			data[layer*model.
			Config.
			Channels:]
		// Activations (l_* is the output of the layer)
		l_ln1 := model.
			Optimizer.
			Activations.
			Layer1Act.
			data[layer*B*T*model.
			Config.
			Channels:]
		l_ln1_mean := model.
			Optimizer.
			Activations.
			LayerNorm1Mean.
			data[layer*B*T:]
		l_ln1_rstd := model.
			Optimizer.
			Activations.
			LayerNorm1Rstd.
			data[layer*B*T:]
		l_qkv := model.
			Optimizer.
			Activations.
			QueryKeyVal.
			data[layer*B*T*3*model.
			Config.
			Channels:]
		l_atty := model.
			Optimizer.
			Activations.
			AttentionInter.
			data[layer*B*T*model.
			Config.
			Channels:]
		l_preatt := model.
			Optimizer.
			Activations.
			PreAttention.
			data[layer*B*model.
			Config.
			NumHeads*T*T:]
		l_att := model.
			Optimizer.
			Activations.
			Attention.
			data[layer*B*model.
			Config.
			NumHeads*T*T:]
		l_attproj := model.
			Optimizer.
			Activations.
			AttentionProj.
			data[layer*B*T*model.
			Config.
			Channels:]
		l_residual2 := model.
			Optimizer.
			Activations.
			Residual2.
			data[layer*B*T*model.
			Config.
			Channels:]
		l_ln2 := model.
			Optimizer.
			Activations.
			LayerNorm2Act.
			data[layer*B*T*model.
			Config.
			Channels:]
		l_ln2_mean := model.
			Optimizer.
			Activations.
			LayerNorm2Mean.
			data[layer*B*T:]
		l_ln2_rstd := model.
			Optimizer.
			Activations.
			LayerNorm2Rstd.
			data[layer*B*T:]
		l_fch := model.
			Optimizer.
			Activations.
			FeedForward.
			data[layer*B*T*4*model.
			Config.
			Channels:]
		l_fch_gelu := model.
			Optimizer.
			Activations.
			FeedForwardGelu.
			data[layer*B*T*4*model.
			Config.
			Channels:]
		l_fcproj := model.
			Optimizer.
			Activations.
			FeedForwardProj.
			data[layer*B*T*model.
			Config.
			Channels:]
		l_residual3 := model.
			Optimizer.
			Activations.
			Residual3.
			data[layer*B*T*model.
			Config.
			Channels:]
		// Normalize layer so that the mean is 0 and the standard deviation is ~1.
		torch.LayernormForward(
			l_ln1,
			l_ln1_mean,
			l_ln1_rstd,
			residual, // residual input
			l_ln1w,   // weight
			l_ln1b,   // bias
			B,
			T,
			model.Config.Channels,
		)

		// Get a layer activation for the model inputs (activations) which are multiplied by the model weights.
		//
		// Input "projection" via linear transformations via the model query/key/value weights into higher dimensionality.
		//
		// l_qkvw = weight = Query Key Val Weights (C * 3C)
		// l_ln1 = inp = layer activations
		// l_qkvb = bias = Query Key Val Bias
		// l_qkv = out = key/query/value matrix
		torch.MatmulForward(
			l_qkv,
			l_ln1,
			l_qkvw,
			l_qkvb,
			B,
			T,
			model.Config.Channels,
			3*model.Config.Channels,
		)
		torch.AttentionForward(
			l_atty,
			l_preatt,
			l_att,
			l_qkv,
			B,
			T,
			model.Config.Channels,
			model.Config.NumHeads,
		)
		// Project the l_atty into another dimension.
		torch.MatmulForward(
			l_attproj,
			l_atty,
			l_attprojw,
			l_attprojb,
			B,
			T,
			model.Config.Channels,
			model.Config.Channels,
		)
		torch.ResidualForward(
			l_residual2,
			residual,
			l_attproj,
			B*T*model.Config.Channels,
		)
		// The weights in this level are the layer 2 activations, which are multiplied with the residual through the above sections
		// This is normalised and everything into layernorm2
		torch.LayernormForward(
			l_ln2,
			l_ln2_mean,
			l_ln2_rstd,
			l_residual2,
			l_ln2w,
			l_ln2b,
			B,
			T,
			model.Config.Channels,
		)
		// Feedforward is just another layer of a multi layer perceptron to make the "higher level" connections.
		torch.MatmulForward(
			l_fch,
			l_ln2,
			l_fcw,
			l_fcb,
			B,
			T,
			model.Config.Channels,
			4*model.Config.Channels,
		)
		torch.GeluForward(
			l_fch_gelu,
			l_fch,
			B*T*4*model.Config.Channels,
		)
		// Squishes the last layer into a smaller dimension so it can be added to the next layer.
		torch.MatmulForward(
			l_fcproj,
			l_fch_gelu,
			l_fcprojw,
			l_fcprojb,
			B,
			T,
			4*model.Config.Channels,
			model.Config.Channels,
		)
		// Set the next residual layer as the output of this layer. This is the l_fcproj + the current layer residual
		torch.ResidualForward(l_residual3, l_residual2, l_fcproj, B*T*model.Config.Channels)
	}
	residual = model.Optimizer.Activations.Residual3.data[(model.Config.NumLayers-1)*B*T*model.Config.Channels:]
	// Normalize the final layer activations to calculate the logits.
	torch.LayernormForward(
		model.Optimizer.Activations.LayerNormFinal.data,
		model.Optimizer.Activations.LayerNormFinalMean.data,
		model.Optimizer.Activations.LayerNormFinalStd.data,
		residual,
		model.Params.LayerFinNormW.data,
		model.Params.LayerFinNormB.data,
		B,
		T,
		model.Config.Channels,
	)
	// Multiply the Word Token embedding by the LayerNormFinal giving the logits.
	torch.MatmulForward(
		model.Optimizer.Activations.Logits.data,
		model.Optimizer.Activations.LayerNormFinal.data,
		model.Params.WordTokEmbed.data,
		nil,
		B,
		T,
		model.Config.Channels,
		model.Config.VocabSize,
	)
	// Softmax the logits to get probabilities over the entire vocabulary
	torch.SoftmaxForward(model.Optimizer.Activations.Probabilities.data, model.Optimizer.Activations.Logits.data, B, T, model.Config.VocabSize)
	// forward the cross-entropy loss function if we have the target tokens
	if len(target) <= 0 {
		model.MeanLoss = -1.0
		return
	}
	// Compare the probabilities for each token and compares it to the target to calculate a loss.
	torch.CrossEntropyForward(
		model.Optimizer.Activations.Losses.data,
		model.Optimizer.Activations.Probabilities.data,
		target,
		B,
		T,
		model.Config.VocabSize,
	)
	// evaluate the mean loss (for convenience)
	var meanLoss float32
	for i := range model.Optimizer.Activations.Losses.data {
		meanLoss += model.Optimizer.Activations.Losses.data[i]
	}
	meanLoss /= float32(B * T)
	model.MeanLoss = meanLoss
}

// ZeroGradient resets the gradients to zero.
func (model *GPT2) ZeroGradient() {
	for i := range model.GradientActivations.Memory {
		model.GradientActivations.Memory[i] = 0.0
	}
	for i := range model.Gradients.Memory {
		model.Gradients.Memory[i] = 0.0
	}
}

// Train trains the model.
// It takes a validation data loader and a training data loader.
// The validation data loader is used to compute the validation loss
// and the training data loader is used to compute the training loss.
func (model *GPT2) Train(valDataloader, trainDataloader *data.DataLoader, B, T int, learningRate, weightDecay, beta1, beta2, eps float32) error {
	fmt.Printf("train dataset num_batches: %d\n", valDataloader.NumBatches)
	genTokens := make([]int32, B*T)
	for step := 0; step <= 40; step++ {
		if step%10 == 0 {
			var valLoss float32
			valDataloader.Reset()
			for i := 0; i < valNumBatches; i++ {
				input, target := valDataloader.NextBatch()
				model.Forward(input, target, B, T)
				valLoss += model.MeanLoss
			}
			valLoss /= float32(valNumBatches)
			fmt.Printf("val loss %f\n", valLoss)
		}
		if step > 0 && step%20 == 0 {
			for i := 0; i < B*T; i++ {
				genTokens[i] = model.Config.EOT
			}
			for t := 1; t < len(genTokens); t++ {
				// Recompute all activations between 0 and t
				model.Forward(genTokens, nil, B, t)
				probabilities := model.Optimizer.Activations.Probabilities.data[(t-1)*model.Config.VocabSize:]
				coin := rand.Float32()
				nextToken2 := sampleMult(probabilities, coin)
				genTokens[t] = rune(nextToken2)
			}
			fmt.Print("generated: ")
			if model.Tokenizer.init {
				str, err := model.Tokenizer.Decode(genTokens)
				if err != nil {
					return err
				}
				fmt.Println(str)
			} else {
				fmt.Println(genTokens)
			}
			for t := 0; t < genMaxLength; t++ {
				if model.Tokenizer.init {

				} else {
					fmt.Printf("%d ", genTokens[t])
				}
			}
			fmt.Println()
		}
		start := time.Now()
		input, targets := trainDataloader.NextBatch()
		model.Forward(input, targets, B, T)
		model.ZeroGradient()
		err := model.Backward()
		if err != nil {
			return fmt.Errorf("failed to backward: %v", err)
		}
		model.Update(learningRate, beta1, beta2, eps, weightDecay, step+1)
		fmt.Printf("step %d: train loss %f (took %v ms)\n", step, model.MeanLoss, time.Since(start))
	}
	return nil
}
