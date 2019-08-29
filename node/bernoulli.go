package node

import (
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

type Bernoulli struct {
	name  string
	value float64 // Gonum's implementation returns a float
	P     Var

	Src *rand.Rand
}

func NewBernoulli(name string, p Var, src *rand.Rand) *Bernoulli {
	defaultValue := 0.0
	newBernoulli := Bernoulli{
		name:  name,
		value: defaultValue,
		P:     p,
		Src:   src,
	}
	return &newBernoulli
}

func (b *Bernoulli) LogProb() float64 {
	dist := distuv.Bernoulli{P: b.P.Value()}
	return dist.LogProb(b.value)
}

func (b *Bernoulli) Rand() float64 {
	dist := distuv.Bernoulli{P: b.P.Value(), Src: b.Src}
	return dist.Rand()
}

func (b *Bernoulli) Name() string {
	return b.name
}

func (b *Bernoulli) Value() float64 {
	return b.value
}

func (b *Bernoulli) SetValue(newValue float64) error {
	roundedVal := math.Round(newValue)
	if roundedVal != 0 && roundedVal != 1 {
		return &OutOfBoundsErr{"A bernoulli-distributed random  variable can only take the values 0 or 1."}
	}
	b.value = newValue

	return nil
}
