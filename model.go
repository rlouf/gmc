package main

import (
	"log"
	"math"

	"github.com/rlouf/gmc/node"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/gonum/stat/samplemv"
)

// A Model contains the information necessary to describe a directed
// probabilistic graphical model and its inference environment.
//
// A probabilistic graphical model (PGM) is a convenient way to represent
// a multivariate joint probability distribution as a graph.
//
// A PGM can contain three types of nodes:
// - Stochastic nodes, which represent a random variables;
// - Deterministic nodes that can be constants, or the result of mathematical
//   transformations applied to the value of other nodes;
// - Observed nodes are nodes in the graph whose value is known.
// An edge represents conditional dependency between the two variables.
//
// Bayesian inference on a PGM consists in infering the possible values for
// the stochastic nodes given the value of the observed nodes and their
// depencies as embedded in the graph.
type Model struct {
	deterministic []node.Var // contains constants and transformed variables
	observed      []node.RandVar
	stochastic    []node.RandVar

	Src *rand.Rand
}

// NewModel creates a new model with sensible defaults.
func NewModel() *Model {
	return &Model{
		Src: rand.New(rand.NewSource(8128)), // obtained from random.org
	}
}

// Observe sets the value of a variable and moves the latter from the
// stochastic set to the observed set.
func (m *Model) Observe(variable node.RandVar, value float64) {
	for i, model_var := range m.stochastic {
		if variable.Name() == model_var.Name() {
			m.stochastic = append(m.stochastic[:i], m.stochastic[i+1:]...)
			m.observed = append(m.observed, model_var)
			model_var.SetValue(value)
			return
		}
	}
	log.Panicf("the variable does not exist: %s", variable.Name())
}

// LogProb computes the log-probability of the graphical model given the
// proposed values for stochastic variables and the fixed value of
// observed variables.
//
// LogProb returns `math.Inf(-1)` when a proposed value is out of bounds.
func (m *Model) LogProb(proposed []float64) float64 {
	if len(proposed) != len(m.stochastic) {
		log.Panicf("needed %d value proposals, got %d", len(m.stochastic), len(proposed))
	}
	for i, value := range proposed {
		err := m.stochastic[i].SetValue(value)
		if err != nil {
			if _, ok := err.(*node.OutOfBoundsErr); ok {
				return math.Inf(-1)
			}
			panic("unexpected error while setting the variable's value")
		}
	}

	var logprob float64
	for _, variable := range m.stochastic {
		logprob += variable.LogProb()
	}
	for _, observed := range m.observed {
		logprob += observed.LogProb()
	}

	return logprob
}

// PriorPredictiveSample generates synthetic values of the observed variables by
// drawing samples from the stochastic variables' prior distribution. It can (should) be used
// to perform prior predictive checks as described in:
//
// "Visualization in Bayesian workflow" (Gabry et al. 2017)
// https://arxiv.org/abs/1709.01449
//
// Since the graphical model is necessarily built from its roots the variables
// are stored in the model in a topological order. Therefore, we only need to
// iterate from left to right to obtain prior samples.
//
// There has been a lot of internal debate about whether sampling methods should
// be functions instead, and I came to the conclusion that these functions can
// only be called in the context of a model, so they are better off as methods
// (besides convenience).
func (m *Model) SamplePriorPredictive(numSamples int) map[string][]float64 {

	samples := make(map[string][]float64)
	for _, o := range m.observed {
		samples[o.Name()] = make([]float64, numSamples, numSamples)
	}

	for i := 0; i < numSamples; i++ {
		for _, v := range m.stochastic {
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
// returns a trace, i.e. a map that links the name of each stochastic variable
// to a slice that contains that values that were sampled for that variable.
//
// Sample is currently very rough arround the edges. For instance, the function
// should accept any struct that implements the `Sample()` function so the
// implementation becomes sampler-independent.
// The function should allow for an automatic search for an appropriate starting
// point, as PyMC3 and Stan do.
// Since all samplers depend on several parameters, we should allow for an auto-tune
// mechanism as in PyMC3 and Stan.
// Ideally, all these would be optional. Maybe adding a Tune(Model) and Init(Model)
// function at the sampler level is our best option?
func (m *Model) Sample(nSamples int, initial []float64, sampler samplemv.MetropolisHastingser) map[string][]float64 {
	if len(initial) != len(m.stochastic) {
		log.Panicf("needed %d initial points, got %d", len(m.stochastic), len(initial))
	}
	sampler.Initial = initial
	sampler.Target = m

	batch := mat.NewDense(nSamples, len(m.stochastic), nil)
	sampler.Sample(batch)
	trace := map[string][]float64{}
	for i := 0; i < nSamples; i++ {
		row := batch.RawRowView(i)
		for j := 0; j < len(m.stochastic); j++ {
			trace[m.stochastic[j].Name()] = append(trace[m.stochastic[j].Name()], row[j])
		}
	}

	return trace
}

// PosteriorPredictiveSample generates synthetic values for the observed variables using
// the posterior samples. This is generally used to perform a posterior predictive check
// on the model as described in:
//
// "Visualization in Bayesian workflow" (Gabry et al. 2017)
// https://arxiv.org/abs/1709.01449
//
// It returns a map from the observed variables' names to a slice of samples.
func (m *Model) SamplePosteriorPredictive(numSamples int, trace map[string][]float64) map[string][]float64 {

	traceSize := 0. // dirty. Include trace size in Trace object
	for _, variable := range m.stochastic {
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
		for _, variable := range m.stochastic {
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

// Normal adds a stochastic variable whose value is normally
// distributed to the model. Returns a pointer to this variable.
func (m *Model) Normal(name string, mu, sigma node.Var) *node.Normal {
	newNormal := node.NewNormal(name, mu, sigma, m.Src)
	if m.IsTaken(name) {
		log.Panicf("variable name is already taken: %s", name)
	}
	m.stochastic = append(m.stochastic, newNormal)
	return newNormal
}

// Beta adds a stochastic variable whose value follows a Beta
// distribution to the model. Returns a pointer to this variable.
func (m *Model) Beta(name string, alpha, beta node.Var) *node.Beta {
	newBeta := node.NewBeta(name, alpha, beta, m.Src)
	if m.IsTaken(name) {
		log.Panicf("variable name is already taken: %s", name)
	}
	m.stochastic = append(m.stochastic, newBeta)
	return newBeta
}

// Bernoulli adds a stochastic variable whose value follows a Bernoulli
// distribution to the model. Returns a pointer to this variable.
func (m *Model) Bernoulli(name string, p node.Var) *node.Bernoulli {
	newBernoulli := node.NewBernoulli(name, p, m.Src)
	if m.IsTaken(name) {
		log.Panicf("variable name is already taken: %s", name)
	}
	m.stochastic = append(m.stochastic, newBernoulli)
	return newBernoulli
}

// Binomial adds a stochastic variable whose value follows a Binomial
// distribution to the model. Returns a pointer to this variable.
func (m *Model) Binomial(name string, N float64, p node.Var) *node.Binomial {
	if N == 0.0 {
		log.Panicf("The number of bernoulli trial must be > 0, got %f", N)
	}
	newBinomial := node.NewBinomial(name, N, p, m.Src)
	if m.IsTaken(name) {
		log.Panicf("variable name is already taken: %s", name)
	}
	m.stochastic = append(m.stochastic, newBinomial)
	return newBinomial
}

// Constant adds a deterministic variable that has a constant value.
func (m *Model) Constant(value float64) node.Var {
	newConst := node.NewConstant(value)
	m.deterministic = append(m.deterministic, newConst)
	return newConst
}

// Sum adds a deterministic node the value of which is the
// sum of the values of the two input nodes to the model.
func (m *Model) Sum(x, y node.Var) node.Var {
	transformed := &node.SumGate{
		X: x,
		Y: y,
	}
	m.deterministic = append(m.deterministic, transformed)
	return transformed
}

// Prod adds a deterministic node the value of which is the
// product of the values of the two input nodes to the model.
func (m *Model) Prod(x, y node.Var) node.Var {
	transformed := &node.ProdGate{
		X: x,
		Y: y,
	}
	m.deterministic = append(m.deterministic, transformed)
	return transformed
}

// Logistic adds to the model a deterministic node the value of which is the
// the logistic transformation of the value of the input node.
func (m *Model) Logistic(x node.Var) node.Var {
	transformed := &node.LogisticGate{
		X: x,
	}
	m.deterministic = append(m.deterministic, transformed)
	return transformed
}

// Logit adds to the model a deterministic node the value of which is the
// the logit transformation of the value of the input node.
func (m *Model) Logit(x node.Var) node.Var {
	transformed := &node.LogitGate{
		X: x,
	}
	m.deterministic = append(m.deterministic, transformed)
	return transformed
}

func (m *Model) Switch(threshold float64, Switch, Left, Right node.Var) node.Var {
	transformed := &node.SwitchGate{
		Threshold: threshold,
		Switch:    Switch,
		Left:      Left,
		Right:     Right,
	}
	m.deterministic = append(m.deterministic, transformed)
	return transformed
}

// IsTaken returns `true` is the name passed as an input has already
// been taken by a node in the graph.
func (m *Model) IsTaken(name string) bool {
	for _, variable := range m.stochastic {
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
