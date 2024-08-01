package model

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Helper function to compare two matrices
func matricesEqual(a, b *mat.Dense, tol float64) bool {
	r1, c1 := a.Dims()
	r2, c2 := b.Dims()
	if r1 != r2 || c1 != c2 {
		return false
	}
	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			if math.Abs(a.At(i, j)-b.At(i, j)) > tol {
				return false
			}
		}
	}
	return true
}

// TestGelu tests the Gelu function.
func TestGelu(t *testing.T) {
	tests := []struct {
		input    *mat.Dense
		expected *mat.Dense
	}{
		{
			input: mat.NewDense(2, 3, []float64{
				0.0, 0.5, 1.0,
				1.5, 2.0, 2.5,
			}),
			expected: mat.NewDense(2, 3, []float64{
				0, 0.34571400982514394, 0.8411919906082768,
				1.3995715769802328, 1.954597694087775, 2.484915733910001,
			}),
		},
	}
	for _, test := range tests {
		result := Gelu(test.input)
		if !matricesEqual(result, test.expected, 1e-5) {
			t.Errorf(
				"Gelu(%v) = %v, expected %v",
				mat.Formatted(test.input),
				mat.Formatted(result),
				mat.Formatted(test.expected),
			)
		}
	}
}

// TestGeluShapes tests the shapes of the Gelu function.
func TestGeluShapes(t *testing.T) {
	tests := []struct {
		input *mat.Dense
	}{
		{
			input: mat.NewDense(1, 1, []float64{0.0}),
		},
		{
			input: mat.NewDense(3, 3, []float64{
				0.0, 0.5, 1.0,
				1.5, 2.0, 2.5,
				-1.0, -0.5, 0.5,
			}),
		},
	}
	for _, test := range tests {
		result := Gelu(test.input)
		rows, cols := result.Dims()
		inputRows, inputCols := test.input.Dims()
		if rows != inputRows || cols != inputCols {
			t.Errorf(
				"Gelu(%v) has shape (%d, %d), expected (%d, %d)",
				mat.Formatted(test.input),
				rows,
				cols,
				inputRows,
				inputCols,
			)
		}
	}
}
