// Package cmd contains the root command for the GPT-2 CLI.
package cmd

import (
	"os"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
)

// rootArgs is the root command arguments.
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
	tokenizerPath string
	datasetPath   string
	seqLength     int
	learningRate  float32
	weightDecay   float32
	beta1         float32
	beta2         float32
	epsilon       float32
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
		log.SetLevel(log.DebugLevel)
		return nil
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
	rootCmd.AddCommand(NewTestCmd())
	rootCmd.AddCommand(NewTrainCommand())
}
