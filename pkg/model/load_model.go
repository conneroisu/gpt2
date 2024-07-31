package model

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/charmbracelet/log"
)

// LoadModel loads the GPT-2 model and its weights from a file.
func LoadModel(modelPath string) (*GPT2Model, error) {
	log.Debugf("Loading model from %s", modelPath)
	defer log.Debugf("Loaded model from %s", modelPath)
	file, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	log.Debugf("Opened model file %s", modelPath)

	decoder := gob.NewDecoder(file)
	log.Debugf("created new decoder")
	log.Debugf("decoding model")
	if err := decoder.Decode(&model); err != nil {
		return nil, fmt.Errorf("error decoding model: %w", err)
	}

	return &model, nil
}

// SaveModel saves the GPT-2 model and its weights to a file.
func SaveModel(model *GPT2Model, modelPath string) error {
	log.Debugf("Saving model to %s", modelPath)
	defer log.Debugf("Saved model to %s", modelPath)
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

