package tiktoken

import (
	"fmt"
	"io"
	"log"
)

// ReadVocab prints the vocabulary to stdout.
func ReadVocab(vocab map[string]int, w io.Writer) {
	if len(vocab) < 256 {
		log.Fatalf(
			"got len=%d, want vocab with at least 256 elements",
			len(vocab),
		)
	}
	w.Write([]byte("vocab: {"))
	for k, v := range vocab {
		// fmt.Printf("  %q: %d\n", k, v)
		w.Write([]byte(fmt.Sprintf("  %q: %d\n", k, v)))
	}
	w.Write([]byte("}\n"))
}
