package model

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

const (
	geluCoeff = 0.044715
)

var sqrt2Pi = math.Sqrt(2 / math.Pi)

// Gelu activation function.
//
// Gaussian Error Linear Units (GELU) activation function.
func Gelu(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	result := mat.NewDense(rows, cols, nil)
	// Precompute constants
	// Apply GELU element-wise
	result.Apply(func(_, _ int, val float64) float64 {
		return 0.5 * val * (1 + math.Tanh(sqrt2Pi*(val+geluCoeff*math.Pow(val, 3))))
	}, x)
	return result
}

// LayerNorm structure.
type LayerNorm struct {
	Weight, Bias    *mat.Dense
	VarianceEpsilon float64
}

// NewLayerNorm returns a new LayerNorm instance.
func NewLayerNorm(hiddenSize int, eps float64) *LayerNorm {
	weight := mat.NewDense(1, hiddenSize, nil)
	for i := 0; i < hiddenSize; i++ {
		weight.Set(0, i, 1)
	}
	bias := mat.NewDense(1, hiddenSize, nil)
	return &LayerNorm{Weight: weight, Bias: bias, VarianceEpsilon: eps}
}

// Forward performs a forward pass on the LayerNorm instance.
func (ln *LayerNorm) Forward(x *mat.Dense) *mat.Dense {
	mean := mat.NewDense(x.RawMatrix().Rows, 1, nil)
	variance := mat.NewDense(x.RawMatrix().Rows, 1, nil)
	for i := 0; i < x.RawMatrix().Rows; i++ {
		mean.Set(i, 0, mat.Sum(x.RowView(i))/float64(x.RawMatrix().Cols))
	}
	for i := 0; i < x.RawMatrix().Rows; i++ {
		var sum float64
		for j := 0; j < x.RawMatrix().Cols; j++ {
			diff := x.At(i, j) - mean.At(i, 0)
			sum += diff * diff
		}
		variance.Set(i, 0, sum/float64(x.RawMatrix().Cols))
	}
	// Standardize
	for i := 0; i < x.RawMatrix().Rows; i++ {
		for j := 0; j < x.RawMatrix().Cols; j++ {
			x.Set(
				i,
				j,
				(x.At(i, j)-mean.At(i, 0))/math.Sqrt(
					variance.At(i, 0)+ln.VarianceEpsilon,
				),
			)
		}
	}
	x.Apply(func(_, j int, v float64) float64 {
		return v*ln.Weight.At(0, j) + ln.Bias.At(0, j)
	}, x)
	return x
}
