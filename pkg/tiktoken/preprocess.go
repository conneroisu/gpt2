package tiktoken

import (
	"strings"
	"unicode"
)

// PreprocessText preprocesses the input text by lowercasing and removing punctuation.
func PreprocessText(text string) string {
	var sb strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsSpace(r) {
			sb.WriteRune(unicode.ToLower(r))
		}
	}
	return sb.String()
}
