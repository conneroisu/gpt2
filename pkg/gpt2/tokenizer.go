package gpt2

import (
	"encoding/binary"
	"fmt"
	"os"
)

// Tokenizer is an interface for tokenizing text.
type Tokenizer interface {
	Decode(tokens []int32) (string, error)
	Encode(text string) ([]int32, error)
}

// GPT2Tokenizer is a tokenizer for the GPT-2 language model.
type GPT2Tokenizer struct {
	vocabSize  uint32
	tokenTable []string
	init       bool
	trie       trie
}

// NewGPT2Tokenizer returns a new GPT2Tokenizer instance.
func NewGPT2Tokenizer(binPath string) (*GPT2Tokenizer, error) {
	return &GPT2Tokenizer{}, nil
}

// NewTokenizer returns a new GPT2Tokenizer instance.
func NewTokenizer(filename string) (GPT2Tokenizer, error) {
	f, err := os.Open(filename)
	if err != nil {
		return GPT2Tokenizer{}, err
	}
	defer f.Close()
	header := make([]uint32, 256)
	if err := binary.Read(f, binary.LittleEndian, header); err != nil {
		return GPT2Tokenizer{}, err
	}
	if header[0] != 20240328 || header[1] != 1 {
		return GPT2Tokenizer{}, fmt.Errorf("incorrect header for tokenizer")
	}
	tok := GPT2Tokenizer{
		vocabSize:  header[2],
		tokenTable: make([]string, header[2]),
		init:       true,
		trie:       newTrie(nil),
	}
	var length byte
	for i := range tok.tokenTable {
		if err := binary.Read(f, binary.LittleEndian, &length); err != nil {
			return tok, err
		}
		if length <= 0 {
			return tok, fmt.Errorf("tokenizer failure")
		}
		tokenBytes := make([]byte, length)
		if err := binary.Read(f, binary.LittleEndian, tokenBytes); err != nil {
			return tok, err
		}
		tok.tokenTable[i] = string(tokenBytes)
		err = tok.trie.Insert(tokenBytes, int32(i))
		if err != nil {
			return tok, err
		}
	}
	return tok, nil
}

// Decode decodes a sequence of tokens into a string.
func (t GPT2Tokenizer) Decode(tokens []int32) (string, error) {
	s := ""
	for _, token := range tokens {
		if token >= int32(len(t.tokenTable)) {
			return "", fmt.Errorf("not valid token")
		}
		if token != GPT2_EOT {
			s += t.tokenTable[token]
		}
	}
	return s, nil
}

// Encode encodes a string into a sequence of tokens.
func (t GPT2Tokenizer) Encode(text string) ([]int32, error) {
	_, tokens := t.trie.Tokenize([]byte(text))
	return tokens, nil
}
