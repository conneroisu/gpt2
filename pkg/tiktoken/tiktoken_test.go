package tiktoken

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestTiktokenTokenize tests the tokenization of text using the cl100k_base.tiktoken vocabulary.
func TestTiktokenTokenize(t *testing.T) {
	r, err := os.Open(filepath.Join("cl100k_base.tiktoken"))
	if err != nil {
		t.Fatal(err)
	}
	vocab, err := LoadTiktokenVocab(r)
	if err != nil {
		t.Fatal(err)
	}
	text := "Dear Charlie, Ball don't lie. They're not."
	toks := Encode(text, vocab, CL100KBaseSplitPattern)
	if len(toks) != 12 {
		t.Errorf("got len %v, want 12", len(toks))
	}
	d := NewDecoder(vocab)
	parts := d.Decode(toks)
	whole := strings.Join(parts, "")
	if whole != text {
		t.Errorf("got whole = %q, not = text", whole)
	}
	text2 := "Also, Charlie, I'm only a fan of the real charlie brown."
	toks2 := Encode(text2, vocab, CL100KBaseSplitPattern)
	if len(toks2) != 16 {
		t.Errorf("got len %v, want 34", len(toks2))
	}
}
