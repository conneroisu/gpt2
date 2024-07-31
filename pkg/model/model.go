package model

import (
	"gonum.org/v1/gonum/mat"
	"encoding/gob"
)

func init() {
	gob.Register(&mat.Dense{})
}

// Config structure.
type Config struct {
	NHead            int     // Number of attention heads.
	NEmb             int     // Embedding dimension.
	VocabSize        int     // Vocabulary size.
	NPositions       int     // Number of positions.
	LayerNormEpsilon float64 // Layer normalization epsilon.
	NLayer           int     // Number of layers.
}

// GPT2Model structure.
type GPT2Model struct {
	NLayer, NEmb, NVocab int        // Number of layers, embedding dimension, and vocabulary size.
	Wte, Wpe             *mat.Dense // Embedding matrices.
	H                    []*Block   // Stack of transformer blocks.
	LnF                  *LayerNorm // Layer normalization layer.
}

// NewGPT2Model returns a new GPT2Model instance.
func NewGPT2Model(config Config) *GPT2Model {
	wte := mat.NewDense(config.VocabSize, config.NEmb, nil)
	wpe := mat.NewDense(config.NPositions, config.NEmb, nil)
	h := make([]*Block, config.NLayer)
	for i := 0; i < config.NLayer; i++ {
		h[i] = NewBlock(config.NPositions, config, true)
	}
	return &GPT2Model{
		NLayer: config.NLayer,
		NEmb:   config.NEmb,
		NVocab: config.VocabSize,
		Wte:    wte,
		Wpe:    wpe,
		H:      h,
		LnF:    NewLayerNorm(config.NEmb, config.LayerNormEpsilon),
	}
}

// Forward performs a forward pass on the GPT2Model instance.
func (g *GPT2Model) Forward(
	inputIDs, positionIDs *mat.Dense,
	past []*mat.Dense,
) (*mat.Dense, [][]*mat.Dense) {
	inputShapeX, _ := inputIDs.Dims()
	// Initialize embedding matrices
	inputsEmbeds := mat.NewDense(inputShapeX, g.NEmb, nil)
	inputsEmbeds.Product(inputIDs, g.Wte)
	// Add input embeddings and position embeddings.
	positionEmbeds := mat.NewDense(inputShapeX, g.NEmb, nil)
	positionEmbeds.Product(positionIDs, g.Wpe)
	// Add input embeddings and position embeddings
	hiddenStates := mat.NewDense(inputShapeX, g.NEmb, nil)
	hiddenStates.Add(inputsEmbeds, positionEmbeds)
	// Initialize presents for each block
	presents := make([][]*mat.Dense, len(g.H))
	for i, block := range g.H {
		var present []*mat.Dense
		hiddenStates, present = block.Forward(hiddenStates, past)
		presents[i] = present
	}
	// Layer normalization on the final hidden states
	hiddenStates = g.LnF.Forward(hiddenStates)
	return hiddenStates, presents
}

