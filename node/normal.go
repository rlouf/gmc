package node

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

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
