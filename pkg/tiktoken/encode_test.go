package tiktoken

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestEncodeBasic(t *testing.T) {
	txt := "i'm blue dabadee dabadam xxxxxx"

	vocab1 := Train(txt, 256, CL100KBaseSplitPattern)
	toks1 := Encode(txt, vocab1, CL100KBaseSplitPattern)

	// Convert the *mat.Dense to a []int
	toks1Slice := matDenseToSlice(toks1)

	for _, tok := range toks1Slice {
		if tok >= 256 {
			t.Errorf("found tok=%v, expect none >= 256", tok)
		}
	}

	vocab2 := Train(txt, 258, CL100KBaseSplitPattern)
	toks2 := Encode(txt, vocab2, CL100KBaseSplitPattern)

	// Convert the *mat.Dense to a []int
	toks2Slice := matDenseToSlice(toks2)

	// should have encoded the 'xx's and 'da'
	tl := len(toks2Slice)
	if vocab2["xx"] != 256 || toks2Slice[tl-1] != 256 || toks2Slice[tl-2] != 256 ||
		toks2Slice[tl-3] != 256 {
		t.Errorf("want last three tokens to be 'xx', got %v", toks2Slice)
	}

	if vocab2["da"] != 257 || toks2Slice[tl-6] != 257 {
		t.Errorf("want token to be 257, got %v", toks2Slice)
	}
}

// Helper function to convert *mat.Dense to []int
func matDenseToSlice(matrix *mat.Dense) []int {
	r, c := matrix.Dims()
	if r != 1 {
		panic("expected single row matrix")
	}

	slice := make([]int, c)
	for i := 0; i < c; i++ {
		slice[i] = int(matrix.At(0, i))
	}
	return slice
}
