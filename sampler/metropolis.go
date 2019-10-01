package sampler

import (
	"log"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/samplemv"
)

type MetropolisHastings struct {
	*samplemv.MetropolisHastingser
	NumVariables int
}

func (m *MetropolisHastings) Run(numSamples int) *mat.Dense {
	if m.Initial == nil {
		log.Panicf("you need to provide initial values to the sampler: run <sampler>.Tune() for automatic initialization, or specify the value of the `Initial` parameter.")
	}

	if len(m.Initial) != m.NumVariables {
		log.Panicf("needed %d initial points, got %d: please change the value of the `Initial` parameter or run the Tune() method", m.NumVariables, len(m.Initial))
	}

	batch := mat.NewDense(numSamples, m.NumVariables, nil)
	m.Sample(batch)

	return batch
}
