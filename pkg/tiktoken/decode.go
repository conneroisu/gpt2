package tiktoken

import (
	"gonum.org/v1/gonum/mat"
)

// Decoder decodes a list of token IDs into a list of text elements that can
// be concatenated together to produce the original text.
type Decoder struct {
	token2str map[int]string
}

// NewDecoder creates a new Decoder from the given vocabulary.
func NewDecoder(vocab map[string]int) *Decoder {
	token2str := make(map[int]string)
	for k, v := range vocab {
		token2str[v] = k
	}
	return &Decoder{
		token2str: token2str,
	}
}

// Decode a matrix of tokens into a list of strings these tokens represent.
func (d *Decoder) Decode(tokens *mat.Dense) []string {
	var s []string
	tokensSlice := matDenseToSlice(tokens)
	for _, t := range tokensSlice {
		s = append(s, d.token2str[t])
	}
	return s
}

// matDenseToSlice converts a matrix to a slice of integers.
func matDenseToSlice(m *mat.Dense) []int {
	rows, cols := m.Dims()
	var s []int
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			s = append(s, int(m.At(i, j)))
		}
	}
	return s
}
