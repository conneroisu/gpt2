package model

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSplitHeads(t *testing.T) {
	// Create a sample input matrix
	inputData := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	input := mat.NewDense(4, 4, inputData)

	// Configuration for attention
	config := Config{
		NHead: 2,
	}
	attention := NewAttention(4, 4, config, true)

	// Test case when k is false
	expectedDataFalse := []float64{
		1, 2, 5, 6, 9, 10, 13, 14,
		3, 4, 7, 8, 11, 12, 15, 16,
	}
	expectedFalse := mat.NewDense(2, 8, expectedDataFalse)
	resultFalse := attention.SplitHeads(input, false)
	if !mat.EqualApprox(resultFalse, expectedFalse, 1e-10) {
		t.Errorf(
			"TestSplitHeads failed for k=false.\nExpected:\n%v\nGot:\n%v",
			mat.Formatted(expectedFalse),
			mat.Formatted(resultFalse),
		)
	}

	// Test case when k is true
	expectedDataTrue := []float64{
		1, 5, 9, 13, 2, 6, 10, 14,
		3, 7, 11, 15, 4, 8, 12, 16,
	}
	expectedTrue := mat.NewDense(2, 8, expectedDataTrue)
	resultTrue := attention.SplitHeads(input, true)
	if !mat.EqualApprox(resultTrue, expectedTrue, 1e-10) {
		t.Errorf(
			"TestSplitHeads failed for k=true.\nExpected:\n%v\nGot:\n%v",
			mat.Formatted(expectedTrue),
			mat.Formatted(resultTrue),
		)
	}
}
