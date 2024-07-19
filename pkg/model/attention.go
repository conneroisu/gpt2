package model

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Attention structure
type Attention struct {
	Bias         *mat.Dense
	NHead        int
	SplitSize    int
	Scale        bool
	CAttn, CProj *Conv1D
}

// NewAttention returns a new Attention instance.
func NewAttention(nx, nCtx int, config Config, scale bool) *Attention {
	nState := nx
	cAttn := NewConv1D(nState*3, nx)
	cProj := NewConv1D(nState, nx)
	bias := mat.NewDense(nCtx, nCtx, nil)
	for i := 0; i < nCtx; i++ {
		for j := 0; j <= i; j++ {
			bias.Set(i, j, 1)
		}
	}
	return &Attention{
		Bias:      bias,
		NHead:     config.NHead,
		SplitSize: nState,
		Scale:     scale,
		CAttn:     cAttn,
		CProj:     cProj,
	}
}

// Attention performs the attention calculation.
//
// It consists of the following steps:
//
// 1. Compute the dot product between q and k^T.
//
// 2. Scale the dot product if necessary.
//
// 3. Apply softmax function to each row of w.
//
// 4. Compute the dot product between the weights (w) and the value (v) matrices.
func (a *Attention) Attention(q, k, v *mat.Dense) *mat.Dense {
	// Step 1: Compute the dot product between q and k^T
	w := mat.NewDense(q.RawMatrix().Rows, k.RawMatrix().Rows, nil)
	w.Product(q, k.T())
	if a.Scale {
		scaleFactor := 1.0 / math.Sqrt(float64(k.RawMatrix().Cols))
		w.Apply(func(_, _ int, v float64) float64 {
			return v * scaleFactor
		}, w)
	}

	// Step 3: Apply softmax function to each row of w
	rows, cols := w.Dims()
	for i := 0; i < rows; i++ {
		row := mat.Row(nil, i, w)
		expSum := 0.0
		for j := 0; j < cols; j++ {
			row[j] = math.Exp(row[j])
			expSum += row[j]
		}
		for j := 0; j < cols; j++ {
			row[j] /= expSum
		}
		w.SetRow(i, row)
	}

	// Step 4: Compute the dot product between the weights (w) and the value (v) matrices
	output := mat.NewDense(q.RawMatrix().Rows, v.RawMatrix().Cols, nil)
	output.Product(w, v)
	return output
}

// Forward performs a forward pass on the Attention instance.
func (a *Attention) Forward(
	x *mat.Dense,
	layerPast []*mat.Dense,
) (*mat.Dense, []*mat.Dense) {
	x = a.CAttn.Forward(x)
	rows, cols := x.Dims()
	query := mat.NewDense(rows, cols/3, x.RawMatrix().Data[:rows*cols/3])
	key := mat.NewDense(
		rows,
		cols/3,
		x.RawMatrix().Data[rows*cols/3:2*rows*cols/3],
	)
	value := mat.NewDense(rows, cols/3, x.RawMatrix().Data[2*rows*cols/3:])
	query = a.SplitHeads(query, false)
	key = a.SplitHeads(key, true)
	value = a.SplitHeads(value, false)
	if layerPast != nil {
		key = mat.NewDense(rows, cols/3+layerPast[0].RawMatrix().Cols, nil)
		key.Stack(layerPast[0], key)
		value = mat.NewDense(rows, cols/3+layerPast[1].RawMatrix().Cols, nil)
		value.Stack(layerPast[1], value)
	}
	present := []*mat.Dense{key, value}
	attnOutput := a.Attention(query, key, value)
	attnOutput = a.MergeHeads(attnOutput)
	attnOutput = a.CProj.Forward(attnOutput)
	return attnOutput, present
}
