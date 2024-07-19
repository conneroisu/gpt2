package model

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Conv1D structure.
//
// Conv1D is a one-dimensional convolutional layer.
type Conv1D struct {
	Weight, Bias *mat.Dense
	Nf, Nx       int
}

// NewConv1D returns a new Conv1D instance.
func NewConv1D(nf, nx int) *Conv1D {
	weight := mat.NewDense(nx, nf, nil)
	for i := 0; i < nx; i++ {
		for j := 0; j < nf; j++ {
			// initialize weights with normal distribution.
			weight.Set(i, j, rand.NormFloat64()*0.02)
		}
	}
	bias := mat.NewDense(1, nf, nil)
	return &Conv1D{Weight: weight, Bias: bias, Nf: nf, Nx: nx}
}

// Forward performs a forward pass on the Conv1D instance.
func (c *Conv1D) Forward(x *mat.Dense) *mat.Dense {
	sizeOutRows, sizeOutCols := x.RawMatrix().Rows, c.Nf
	output := mat.NewDense(sizeOutRows, sizeOutCols, nil)
	output.Product(x, c.Weight)
	output.Apply(func(_, j int, v float64) float64 {
		return v + c.Bias.At(0, j)
	}, output)
	return output
}
