// Package cmd contains the root command for the GPT-2 CLI.
package cmd

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
	"your_project/pkg/model"
	"your_project/pkg/tiktoken"
)

type rootArgs struct {
	text          string
	verbose       bool
	nSamples      int
	unconditional bool
	batchSize     int
	length        int
	temperature   float64
	topK          int
	seed          int
	modelPath     string
}

// RootArgs is the root command arguments.
var RootArgs rootArgs

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "gpt2",
	Short: "A CLI for the GPT-2 language model",
	Long: `
A CLI for the GPT-2 language model.

Allows you to interact with the GPT-2 language model.
	`,
	PreRunE: func(cmd *cobra.Command, _ []string) error {
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
			StringVarP(&RootArgs.modelPath, "model-path", "m", "model.gob", "Path to the model file")
		log.SetLevel(log.DebugLevel)
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		// Load the model
		model, err := model.LoadModel(RootArgs.modelPath)
		if err != nil {
			log.Fatalf("Failed to load model: %v", err)
		}

		// Generate text
		generatedText := model.GenerateText(model, RootArgs.text, RootArgs.length, RootArgs.temperature)
		fmt.Println(generatedText)
	},
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	// Here you will define your flags and configuration settings.
	// Cobra supports persistent flags, which, if defined here,
	// will be global for your application.

	// rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.gpt2.yaml)")

	// Cobra also supports local flags, which will only run
	// when this action is called directly.
	rootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}
