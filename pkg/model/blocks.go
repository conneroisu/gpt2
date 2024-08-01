package model

import "gonum.org/v1/gonum/mat"

// Block structure
type Block struct {
	Ln1, Ln2 *LayerNorm
	Attn     *Attention
	Mlp      *MLP
}

// NewBlock returns a new Block instance.
func NewBlock(nCtx int, config Config, scale bool) *Block {
	return &Block{
		Ln1:  NewLayerNorm(config.NEmb, config.LayerNormEpsilon),
		Attn: NewAttention(config.NEmb, nCtx, config, scale),
		Ln2:  NewLayerNorm(config.NEmb, config.LayerNormEpsilon),
		Mlp:  NewMLP(4*config.NEmb, config),
	}
}

// Forward performs a forward pass on the Block instance.
func (b *Block) Forward(
	x *mat.Dense,
	layerPast []*mat.Dense,
) (*mat.Dense, []*mat.Dense) {
	a, present := b.Attn.Forward(b.Ln1.Forward(x), layerPast)
	x.Add(x, a)
	m := b.Mlp.Forward(b.Ln2.Forward(x))
	x.Add(x, m)
	return x, present
}
