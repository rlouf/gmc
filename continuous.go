// Probability distributions with a continuous support.
package main

import (
	"fmt"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

type OutOfBoundsErr struct {
	msg string
}

func (o *OutOfBoundsErr) Error() string {
	return o.msg
}

// ------------------------------
// Normal distribution
// -----------------------------

type Normal struct {
	name  string
	value float64
	Mu    Var
	Sigma Var

	Src *rand.Rand
}

func NewNormal(name string, mu, sigma Var, src *rand.Rand) *Normal {
	defaultValue := mu.Value()
	newNormal := Normal{
		name:  name,
		value: defaultValue,
		Mu:    mu,
		Sigma: sigma,
		Src:   src,
	}
	return &newNormal
}

func (n *Normal) LogProb() float64 {
	dist := distuv.Normal{Mu: n.Mu.Value(), Sigma: n.Sigma.Value()}
	return dist.LogProb(n.value)
}

func (n *Normal) Rand() float64 {
	dist := distuv.Normal{Mu: n.Mu.Value(), Sigma: n.Sigma.Value(), Src: n.Src}
	return dist.Rand()
}

func (n *Normal) Name() string {
	return n.name
}

func (n *Normal) Value() float64 {
	return n.value
}

func (n *Normal) SetValue(newValue float64) error {
	n.value = newValue
	return nil
}

// ----------------------
// Beta distribution
// ----------------------

type Beta struct {
	name  string
	value float64
	Alpha Var
	Beta  Var

	Src *rand.Rand
}

func newBeta(name string, alpha, beta Var, src *rand.Rand) *Beta {
	defaultValue := 0.5
	newBeta := Beta{
		name:  name,
		value: defaultValue,
		Alpha: alpha,
		Beta:  beta,
		Src:   src,
	}
	return &newBeta
}

func (b *Beta) LogProb() float64 {
	dist := distuv.Beta{Alpha: b.Alpha.Value(), Beta: b.Beta.Value()}
	return dist.LogProb(b.Value())
}

func (b *Beta) Rand() float64 {
	beta := distuv.Beta{Alpha: b.Alpha.Value(), Beta: b.Beta.Value(), Src: b.Src}
	return beta.Rand()
}

func (b *Beta) Value() float64 {
	return b.value
}

func (b *Beta) Name() string {
	return b.name
}

func (b *Beta) SetValue(newValue float64) error {

	if newValue < 0 || newValue > 1 {
		return &OutOfBoundsErr{fmt.Sprintf("Beta is defined on [0,1], got value %f", newValue)}
	}
	b.value = newValue

	return nil
}
