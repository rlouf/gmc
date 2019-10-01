package main

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/samplemv"

	"github.com/rlouf/gmc/sampler"
)

func NewMetropolisHastingsSampler(model *Model) *sampler.MetropolisHastings {
	sigmaSym := mat.NewSymDense(1, []float64{0.05})
	proposal, _ := samplemv.NewProposalNormal(sigmaSym, model.Src)

	sampler := sampler.MetropolisHastings{
		MetropolisHastingser: &samplemv.MetropolisHastingser{
			BurnIn:   1000,
			Proposal: proposal,
			Src:      model.Src,
			Target:   model},
		NumVariables: len(model.stochastic),
	}

	return &sampler
}
