package model

import "gonum.org/v1/gonum/mat"

// MLP structure
type MLP struct {
	CFc, CProj *Conv1D
	Act        func(*mat.Dense) *mat.Dense
}

// NewMLP returns a new MLP instance.
func NewMLP(nState int, config Config) *MLP {
	return &MLP{
		CFc:   NewConv1D(nState, config.NEmb),
		CProj: NewConv1D(config.NEmb, nState),
		Act:   Gelu,
	}
}

// Forward performs a forward pass on the MLP instance.
func (m *MLP) Forward(x *mat.Dense) *mat.Dense {
	h := m.Act(m.CFc.Forward(x))
	return m.CProj.Forward(h)
}
