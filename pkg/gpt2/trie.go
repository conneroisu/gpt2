package gpt2

import "fmt"

// trie is a trie data structure.
type trie struct {
	children map[byte]*trie
	data     int32
	end      bool
	key      byte
}

// newTrie creates a new trie.
func newTrie(data []string) trie {
	t := trie{
		children: map[byte]*trie{},
		end:      false,
	}
	for i, word := range data {
		err := t.Insert([]byte(word), int32(i))
		if err != nil {
			panic(err)
		}
	}
	return t
}

// Insert inserts a word into the trie.
func (t *trie) Insert(word []byte, data int32) error {
	cur := t
	if len(word) == 0 {
		return fmt.Errorf("zero length word not supported")
	}
	var index byte
	for i := 0; i < len(word); i++ {
		index = word[i] // 00: 0
		if cur.children[index] == nil {
			cur.children[index] = &trie{
				children: map[byte]*trie{},
			}
		}
		cur = cur.children[index]
	}
	cur.end = true
	cur.data = data
	cur.key = index
	return nil
}

// Tokenize tokenizes a string.
func (t *trie) Tokenize(input []byte) ([][]byte, []int32) {
	var cur = t
	var token = GPT2_EOT
	endIdx, next := 1, 0
	split, tokens := make([][]byte, 0), make([]int32, 0)
	for len(input) != 0 {
		switch {
		case next == len(input), cur.children[input[next]] == nil:
			split = append(split, input[:endIdx])
			tokens = append(tokens, token)
			input = input[endIdx:]
			token = GPT2_EOT
			cur = t
			next = 0
			endIdx = 1
		default:
			cur = cur.children[input[next]]
			next += 1
			if cur.end {
				endIdx = next
				token = cur.data
			}
		}
	}
	return split, tokens
}
