package cmd

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/conneroisu/gpt2/pkg/data"
	"github.com/conneroisu/gpt2/pkg/gpt2"
	"github.com/spf13/cobra"
)

// NewTrainCommand returns a new train command.
func NewTrainCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "train",
		Short: "Train a model",
		Long: `
A CLI for the GPT-2 language model.

Allows you to interact with the GPT-2 language model.
	`,
		RunE: func(cmd *cobra.Command, args []string) error {
			model, err := gpt2.LoadModel(RootArgs.modelPath, RootArgs.tokenizerPath)
			if err != nil {
				log.Fatalf("Failed to load model: %v", err)
			}
			loader, err := data.NewDataLoader(
				RootArgs.datasetPath,
				RootArgs.batchSize,
				RootArgs.seqLength,
			)
			if err != nil {
				log.Fatalf("Failed to load data loader: %v", err)
			}
			validationLoader, err := data.NewDataLoader(
				RootArgs.datasetPath,
				RootArgs.batchSize,
				RootArgs.seqLength,
			)
			if err != nil {
				log.Fatalf("Failed to load data loader: %v", err)
			}
			err = model.Train(
				loader,
				validationLoader,
				RootArgs.batchSize,
				RootArgs.seqLength,
				RootArgs.learningRate,
				RootArgs.weightDecay,
				RootArgs.beta1,
				RootArgs.beta2,
				RootArgs.epsilon,
			)
			if err != nil {
				return fmt.Errorf("failed to train model: %v", err)
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
		StringVarP(&RootArgs.modelPath, "model-path", "m", "model.ckpt", "Path to the model file")
	cmd.PersistentFlags().
		StringVarP(&RootArgs.tokenizerPath, "tokenizer-path", "p", "tokenizer.bin", "Path to the tokenizer file")
	cmd.PersistentFlags().
		StringVarP(&RootArgs.datasetPath, "dataset-path", "d", "dataset.bin", "Path to the dataset file")
	cmd.PersistentFlags().
		IntVarP(&RootArgs.seqLength, "seq-length", "l", 64, "Sequence length")
	cmd.PersistentFlags().
		Float32VarP(&RootArgs.learningRate, "learning-rate", "r", 1e-4, "Learning rate")
	cmd.PersistentFlags().
		Float32VarP(&RootArgs.weightDecay, "weight-decay", "w", 0.0, "Weight decay")
	cmd.PersistentFlags().
		Float32VarP(&RootArgs.beta1, "beta1", "b1", 0.9, "Beta1")
	cmd.PersistentFlags().
		Float32VarP(&RootArgs.beta2, "beta2", "b2", 0.999, "Beta2")
	cmd.PersistentFlags().
		Float32VarP(&RootArgs.epsilon, "epsilon", "e", 1e-8, "Epsilon")
	return cmd
}
