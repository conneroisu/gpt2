package data

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const Int32ByteLen = 4

// Loader is an interface for data loaders.
type Loader interface {
	NextBatch() ([]int32, []int32, error)
	Reset()
}

// DataLoader is a struct that represents a data loader for the GPT-2 model.
type DataLoader struct {
	batchSize  int
	seqLength  int
	curPos     int64
	fileSize   int64
	NumBatches int
	data       []int32
}

// NewDataLoader returns a new DataLoader instance.
func NewDataLoader(filename string, batchSize, seqLength int) (*DataLoader, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	data, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}
	size := len(data)
	if size < (batchSize*seqLength+1)*Int32ByteLen {
		return nil, fmt.Errorf("file size is too small for the batch size and sequence length")
	}
	loader := &DataLoader{
		batchSize:  batchSize,
		seqLength:  seqLength,
		NumBatches: size / (batchSize * seqLength * Int32ByteLen),
		data:       make([]int32, size/Int32ByteLen),
		fileSize:   int64(size / Int32ByteLen),
	}
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, loader.data); err != nil {
		return nil, err
	}
	return loader, nil
}

// Reset resets the loader to the beginning of the file.
func (loader *DataLoader) Reset() {
	loader.curPos = 0
}

// NextBatch returns the next batch of data.
func (loader *DataLoader) NextBatch() ([]int32, []int32) {
	nextPos := loader.curPos + int64(loader.batchSize*loader.seqLength)
	if nextPos+1 > loader.fileSize {
		loader.Reset()
		nextPos = loader.curPos + int64(loader.batchSize*loader.seqLength)
	}
	inputs := loader.data[loader.curPos:nextPos]
	targets := loader.data[loader.curPos+1 : nextPos+1]
	loader.curPos = nextPos
	return inputs, targets
}
