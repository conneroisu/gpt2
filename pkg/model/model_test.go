package model

import (
	"os"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLoadModel(t *testing.T) {
	// Create a temporary file to save the model
	file, err := os.CreateTemp("", "model_test")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(file.Name())

	// Create a sample model
	config := Config{
		NHead:            12,
		NEmb:             768,
		VocabSize:        50257,
		NPositions:       1024,
		LayerNormEpsilon: 1e-5,
		NLayer:           12,
	}
	model := NewGPT2Model(config)

	// Save the model to the temporary file
	if err := SaveModel(model, file.Name()); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	// Load the model from the temporary file
	loadedModel, err := LoadModel(file.Name())
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Compare the original and loaded models
	if !mat.Equal(model.Wte, loadedModel.Wte) {
		t.Errorf("Wte matrices are not equal")
	}
	if !mat.Equal(model.Wpe, loadedModel.Wpe) {
		t.Errorf("Wpe matrices are not equal")
	}
	if len(model.H) != len(loadedModel.H) {
		t.Errorf("Number of blocks are not equal")
	}
	for i := range model.H {
		if !mat.Equal(model.H[i].Ln1.Weight, loadedModel.H[i].Ln1.Weight) {
			t.Errorf("Block %d Ln1 weights are not equal", i)
		}
		if !mat.Equal(model.H[i].Ln2.Weight, loadedModel.H[i].Ln2.Weight) {
			t.Errorf("Block %d Ln2 weights are not equal", i)
		}
	}
	if !mat.Equal(model.LnF.Weight, loadedModel.LnF.Weight) {
		t.Errorf("LnF weights are not equal")
	}
}

func TestSaveModel(t *testing.T) {
	// Create a temporary file to save the model
	file, err := os.CreateTemp("", "model_test")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(file.Name())

	// Create a sample model
	config := Config{
		NHead:            12,
		NEmb:             768,
		VocabSize:        50257,
		NPositions:       1024,
		LayerNormEpsilon: 1e-5,
		NLayer:           12,
	}
	model := NewGPT2Model(config)

	// Save the model to the temporary file
	if err := SaveModel(model, file.Name()); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	// Load the model from the temporary file
	loadedModel, err := LoadModel(file.Name())
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Compare the original and loaded models
	if !mat.Equal(model.Wte, loadedModel.Wte) {
		t.Errorf("Wte matrices are not equal")
	}
	if !mat.Equal(model.Wpe, loadedModel.Wpe) {
		t.Errorf("Wpe matrices are not equal")
	}
	if len(model.H) != len(loadedModel.H) {
		t.Errorf("Number of blocks are not equal")
	}
	for i := range model.H {
		if !mat.Equal(model.H[i].Ln1.Weight, loadedModel.H[i].Ln1.Weight) {
			t.Errorf("Block %d Ln1 weights are not equal", i)
		}
		if !mat.Equal(model.H[i].Ln2.Weight, loadedModel.H[i].Ln2.Weight) {
			t.Errorf("Block %d Ln2 weights are not equal", i)
		}
	}
	if !mat.Equal(model.LnF.Weight, loadedModel.LnF.Weight) {
		t.Errorf("LnF weights are not equal")
	}
}
