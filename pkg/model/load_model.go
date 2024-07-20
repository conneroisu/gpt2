package model

import (
	"encoding/gob"
	"os"

	"gonum.org/v1/gonum/mat"
)

// LoadModel loads the GPT-2 model and its weights from a file.
func LoadModel(modelPath string) (*GPT2Model, error) {
	file, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var model GPT2Model
	if err := decoder.Decode(&model); err != nil {
		return nil, err
	}

	return &model, nil
}

// SaveModel saves the GPT-2 model and its weights to a file.
func SaveModel(model *GPT2Model, modelPath string) error {
	file, err := os.Create(modelPath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		return err
	}

	return nil
}
