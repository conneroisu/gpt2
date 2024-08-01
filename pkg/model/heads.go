package model

import "gonum.org/v1/gonum/mat"

// GPT2LMHead structure
type GPT2LMHead struct {
	Decoder *mat.Dense
	NEmb    int
}

// MergeHeads merges the heads of a attention matrix.
func (a *Attention) MergeHeads(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	newRows := rows * a.NHead
	newCols := cols / a.NHead
	result := mat.NewDense(newRows, newCols, nil)
	for i := 0; i < a.NHead; i++ {
		for j := 0; j < rows; j++ {
			for k := 0; k < cols/a.NHead; k++ {
				result.Set(i*rows+j, k, x.At(j, i*cols/a.NHead+k))
			}
		}
	}
	return result
}

// SplitHeads splits the heads of an attention matrix.
//
// k is the parameter that controls the permutation of the dimensions.
//
// If k is true, the matrix is permuted as (batch, seq_length, n_head, head_features) -> (batch, n_head, seq_length, head_features).
// If k is false, the matrix is permuted as (batch, seq_length, n_head, head_features) -> (batch, n_head, head_features, seq_length).
func (a *Attention) SplitHeads(x *mat.Dense, k bool) *mat.Dense {
	rows, cols := x.Dims()
	headSize := cols / a.NHead
	newRows := a.NHead
	newCols := rows * headSize

	result := mat.NewDense(newRows, newCols, nil)

	if !k {
		// Permute (batch, seq_length, n_head, head_features) -> (batch, n_head, seq_length, head_features)
		for i := 0; i < rows; i++ {
			for j := 0; j < a.NHead; j++ {
				for l := 0; l < headSize; l++ {
					result.Set(j, i*headSize+l, x.At(i, j*headSize+l))
				}
			}
		}
	} else {
		// Permute (batch, seq_length, n_head, head_features) -> (batch, n_head, head_features, seq_length)
		for i := 0; i < rows; i++ {
			for j := 0; j < a.NHead; j++ {
				for l := 0; l < headSize; l++ {
					result.Set(j, l*rows+i, x.At(i, j*headSize+l))
				}
			}
		}
	}

	return result
}
