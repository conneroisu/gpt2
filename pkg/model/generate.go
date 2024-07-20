package model

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

// GenerateText generates text using the GPT-2 model.
func GenerateText(model *GPT2Model, inputText string, length int, temperature float64) string {
	// Preprocess the input text
	preprocessedText := PreprocessText(inputText)

	// Encode the input text
	vocab := LoadVocab() // Assume LoadVocab is a function that loads the vocabulary
	inputIDs := Encode(preprocessedText, vocab, CL100KBaseSplitPattern)

	// Initialize position IDs
	positionIDs := mat.NewDense(len(inputIDs), 1, nil)
	for i := 0; i < len(inputIDs); i++ {
		positionIDs.Set(i, 0, float64(i))
	}

	// Generate text
	var generatedTokens []int
	for i := 0; i < length; i++ {
		hiddenStates, _ := model.Forward(mat.NewDense(len(inputIDs), 1, inputIDs), positionIDs, nil)
		logits := hiddenStates.RawMatrix().Data
		nextToken := sampleNextToken(logits, temperature)
		generatedTokens = append(generatedTokens, nextToken)
		inputIDs = append(inputIDs, nextToken)
		positionIDs = append(positionIDs, float64(len(inputIDs)-1))
	}

	// Decode the generated tokens
	decoder := NewDecoder(vocab)
	generatedText := strings.Join(decoder.Decode(generatedTokens), "")

	return generatedText
}

// sampleNextToken samples the next token from the logits using the given temperature.
func sampleNextToken(logits []float64, temperature float64) int {
	// Apply temperature
	for i := range logits {
		logits[i] /= temperature
	}

	// Compute probabilities
	var sum float64
	for _, logit := range logits {
		sum += math.Exp(logit)
	}
	probs := make([]float64, len(logits))
	for i, logit := range logits {
		probs[i] = math.Exp(logit) / sum
	}

	// Sample from the probabilities
	r := rand.Float64()
	for i, prob := range probs {
		r -= prob
		if r <= 0 {
			return i
		}
	}

	return len(probs) - 1
}
