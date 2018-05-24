package main

import (
	"log"
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/gonum/stat/samplemv"
)

// A Model contains the information necessary to describe a directed
// probabilistic graphical model and its inference environment.
type Model struct {
	static    []Var // contains constants and transformed variables
	observed  []RandVar
	variables []RandVar

	Src *rand.Rand // random number source used for sampling
}

// NewModel initializes a new model with sensible defaults.
// The seed for the random number generator was obtained from random.org
func NewModel() *Model {
	return &Model{
		Src: rand.New(rand.NewSource(8128)),
	}
}

// Observe sets the value of a variable and adds it to the observed set. This
// prevents its value from being changed during the sampling process.
// Observed variables set empirical constraints on the model.
func (m *Model) Observe(variable RandVar, value float64) {
	for i, model_var := range m.variables {
		if variable.Name() == model_var.Name() {
			m.variables = append(m.variables[:i], m.variables[i+1:]...)
			m.observed = append(m.observed, model_var)
			model_var.SetValue(value)
			return
		}
	}
	log.Panicf("the variable does not exist: %s", variable.Name())
}

// LogProb computes the log-probability of the graphical model given the values
// of the observed variables, and a set of proposed values for the random
// variables.
// LogProb returns `math.Inf(-1)` when a proposed value is out of bounds.
func (m *Model) LogProb(proposed []float64) float64 {
	if len(proposed) != len(m.variables) {
		log.Panicf("needed %d value proposals, got %d", len(m.variables), len(proposed))
	}
	for i, value := range proposed {
		err := m.variables[i].SetValue(value)
		if err != nil {
			if _, ok := err.(*OutOfBoundsErr); ok {
				return math.Inf(-1)
			}
			panic("unexpected error while setting the variable's value")
		}
	}

	var logprob float64
	for _, variable := range m.variables {
		logprob += variable.LogProb()
	}
	for _, observed := range m.observed {
		logprob += observed.LogProb()
	}

	return logprob
}

// PriorPredictiveSample generates samples for each variable in the graphical model. It can be used
// to perform prior predictive checks as described in:
//
// "Visualization in Bayesian workflow" (Gabry et al. 2017)
// https://arxiv.org/abs/1709.01449
//
// Since we necessarily build the graphical model from the roots, the variables
// are stored in the model in a topological order. Therefore, we only need to iterate
// from left to right to obtain samples from the graphical model.
func (m *Model) SamplePriorPredictive(numSamples int) map[string][]float64 {

	samples := make(map[string][]float64)
	for _, o := range m.observed {
		samples[o.Name()] = make([]float64, numSamples, numSamples)
	}

	for i := 0; i < numSamples; i++ {
		for _, v := range m.variables {
			v.SetValue(v.Rand())
		}
		for _, o := range m.observed {
			name := o.Name()
			samples[name][i] = o.Rand()
		}
	}

	return samples
}

// Sample generates samples from the posterior distribution of the model. It
// returns a trace, i.e. a map from variable names to a slice of samples from
// their posterior distribution.
func (m *Model) Sample(nSamples int, initial []float64, sampler samplemv.MetropolisHastingser) map[string][]float64 {
	if len(initial) != len(m.variables) {
		log.Panicf("needed %d initial points, got %d", len(m.variables), len(initial))
	}
	sampler.Initial = initial
	sampler.Target = m

	batch := mat.NewDense(nSamples, len(m.variables), nil)
	sampler.Sample(batch)
	trace := map[string][]float64{}
	for i := 0; i < nSamples; i++ {
		row := batch.RawRowView(i)
		for j := 0; j < len(m.variables); j++ {
			trace[m.variables[j].Name()] = append(trace[m.variables[j].Name()], row[j])
		}
	}

	return trace
}

// PosteriorPredictiveSample generates values for the observed variables using
// the samples of the model's posterior distribution. This can be used as part
// of the posterior predictive check described in:
//
// "Visualization in Bayesian workflow" (Gabry et al. 2017)
// https://arxiv.org/abs/1709.01449
//
// It returns a map from the observed variables' names to a slice of samples.
func (m *Model) SamplePosteriorPredictive(numSamples int, trace map[string][]float64) map[string][]float64 {

	traceSize := 0. // dirty. Include trace size in Trace object
	for _, variable := range m.variables {
		if _, ok := trace[variable.Name()]; !ok {
			log.Panicf("The trace is missing variable %s", variable.Name())
		}
		traceSize = float64(len(trace[variable.Name()]))
	}

	samples := make(map[string][]float64)
	for _, o := range m.observed {
		samples[o.Name()] = make([]float64, numSamples, numSamples)
	}

	// We choose one sample from the posterior distribution, set the values
	// of variables and then generate a sample the observed variables
	sampler := distuv.Uniform{Min: 0, Max: traceSize - 1, Src: m.Src}
	var name string
	for i := 0; i < numSamples; i++ {
		loc := int(math.Round(sampler.Rand()))
		for _, variable := range m.variables {
			name = variable.Name()
			variable.SetValue(trace[name][loc])
		}
		for _, observed := range m.observed {
			name = observed.Name()
			samples[name][i] = observed.Rand()
		}
	}

	return samples
}

// The following functions allow to add the variables defined in
// `continuous.go`, `discrete.go` and `static.go` to the model.
func (m *Model) Normal(name string, mu, sigma Var) *Normal {
	newNormal := NewNormal(name, mu, sigma, m.Src)
	if m.IsTaken(name) {
		log.Panicf("variable name is already taken: %s", name)
	}
	m.variables = append(m.variables, newNormal)
	return newNormal
}

func (m *Model) Beta(name string, alpha, beta Var) *Beta {
	newBeta := newBeta(name, alpha, beta, m.Src)
	if m.IsTaken(name) {
		log.Panicf("variable name is already taken: %s", name)
	}
	m.variables = append(m.variables, newBeta)
	return newBeta
}

// Discrete
func (m *Model) Bernoulli(name string, p Var) *Bernoulli {
	newBernoulli := NewBernoulli(name, p, m.Src)
	if m.IsTaken(name) {
		log.Panicf("variable name is already taken: %s", name)
	}
	m.variables = append(m.variables, newBernoulli)
	return newBernoulli
}

func (m *Model) Binomial(name string, N float64, p Var) *Binomial {
	if N == 0.0 {
		log.Panicf("The number of bernoulli trial must be > 0, got %f", N)
	}
	newBinomial := NewBinomial(name, N, p, m.Src)
	if m.IsTaken(name) {
		log.Panicf("variable name is already taken: %s", name)
	}
	m.variables = append(m.variables, newBinomial)
	return newBinomial
}

// Constant
func (m *Model) Constant(value float64) Var {
	newConst := &Constant{value: value}
	m.static = append(m.static, newConst)
	return newConst
}

// Transformations
func (m *Model) Sum(x, y Var) Var {
	transformed := &SumGate{
		X: x,
		Y: y,
	}
	m.static = append(m.static, transformed)
	return transformed
}

func (m *Model) Prod(x, y Var) Var {
	transformed := &ProdGate{
		X: x,
		Y: y,
	}
	m.static = append(m.static, transformed)
	return transformed
}

func (m *Model) Logistic(x Var) Var {
	transformed := &LogisticGate{
		X: x,
	}
	m.static = append(m.static, transformed)
	return transformed
}

func (m *Model) Logit(x Var) Var {
	transformed := &LogitGate{
		X: x,
	}
	m.static = append(m.static, transformed)
	return transformed
}

func (m *Model) Switch(threshold float64, Switch, Left, Right Var) Var {
	transformed := &SwitchGate{
		threshold: threshold,
		Switch:    Switch,
		Left:      Left,
		Right:     Right,
	}
	m.static = append(m.static, transformed)
	return transformed
}

func (m *Model) IsTaken(name string) bool {
	for _, variable := range m.variables {
		if variable.Name() == name {
			return true
		}
	}
	for _, observed := range m.observed {
		if observed.Name() == name {
			return true
		}
	}
	return false
}
