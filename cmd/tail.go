package cmd

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/conneroisu/gpt2/pkg/gpt2"
	"github.com/conneroisu/gpt2/pkg/torch"
	"github.com/spf13/cobra"
)

// NewTestCmd returns a new cobra.Command for the chat command.
func NewTestCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "test",
		Short: "A CLI for the GPT-2 language model",
		Long: `
A CLI for the GPT-2 language model.

Allows you to interact with the GPT-2 language model.
	`,
		RunE: func(cmd *cobra.Command, args []string) error {
			model, err := gpt2.LoadModel(RootArgs.modelPath, RootArgs.tokenizerPath)
			if err != nil {
				log.Fatalf("Failed to load model: %v", err)
			}

			B, T := int(RootArgs.batchSize), int(RootArgs.length)
			x, y := make([]int32, B*T), make([]int32, B*T)
			var losses []float32
			for step := 0; step < 10; step++ {
				start := time.Now()
				model.Forward(x, y, int(B), int(T))
				model.ZeroGradient()
				if err := model.Backward(); err != nil {
					log.Fatal(err)
				}
				elapsed := time.Now().Sub(start)
				model.Update(1e-4, 0.9, 0.999, 1e-8, 0.01, step+1)
				fmt.Printf("step %d: loss %f (took %v)\n", step, model.MeanLoss, elapsed)
				losses = append(losses, model.MeanLoss)
			}
			// expected losses are as follows, from Python
			expectedLosses := []float32{
				5.270007133483887,
				4.059706687927246,
				3.3751230239868164,
				2.8007826805114746,
				2.315382242202759,
				1.8490285873413086,
				1.3946564197540283,
				0.9991465210914612,
				0.6240804195404053,
				0.37651097774505615,
			}
			for i := range expectedLosses {
				if torch.Abs(losses[i]-expectedLosses[i]) >= 1e-2 {
					fmt.Printf("LOSS MISMATCH AT STEP %d: %f %f\n", i, losses[i], expectedLosses[i])
					return fmt.Errorf("loss mismatch at step %d", i)
				} else {
					fmt.Printf("loss ok at step %d: %f %f\n", i, losses[i], expectedLosses[i])
				}
			}
			return nil
		},
	}

	cmd.PersistentFlags().
		StringVarP(&RootArgs.text, "text", "t", "", "Text to generate")
	cmd.PersistentFlags().
		BoolVarP(&RootArgs.verbose, "verbose", "v", false, "Verbose output")
	cmd.PersistentFlags().
		IntVarP(&RootArgs.nSamples, "n-samples", "n", 0, "Number of samples to generate")
	cmd.PersistentFlags().
		BoolVarP(&RootArgs.unconditional, "unconditional", "u", false, "Unconditional generation")
	cmd.PersistentFlags().
		IntVarP(&RootArgs.batchSize, "batch-size", "b", 1, "Batch size")
	cmd.PersistentFlags().
		IntVarP(&RootArgs.length, "length", "l", 0, "Length of generated text")
	cmd.PersistentFlags().
		Float64VarP(&RootArgs.temperature, "temperature", "T", 1.0, "Temperature")
	cmd.PersistentFlags().
		IntVarP(&RootArgs.topK, "top-k", "k", 0, "Top-k sampling")
	cmd.PersistentFlags().
		IntVarP(&RootArgs.seed, "seed", "s", rand.Intn(2147483647), "Seed for random number generator")
	cmd.PersistentFlags().
		StringVarP(&RootArgs.modelPath, "model-path", "m", "model.safetensors", "Path to the model file")
	cmd.PersistentFlags().
		StringVarP(&RootArgs.tokenizerPath, "tokenizer-path", "t", "tokenizer.json", "Path to the tokenizer file")
	return cmd
}
