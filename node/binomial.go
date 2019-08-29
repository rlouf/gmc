package node

import (
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

type Binomial struct {
	name  string
	value float64
	N     float64 // N is the total number of Bernoulli trials > 0
	P     Var

	Src *rand.Rand
}

func NewBinomial(name string, N float64, p Var, src *rand.Rand) *Binomial {
	defaultValue := N * p.Value()
	newBinomial := Binomial{
		name:  name,
		value: defaultValue,
		N:     N,
		P:     p,
		Src:   src,
	}
	return &newBinomial
}

func (b *Binomial) LogProb() float64 {
	dist := distuv.Binomial{N: b.N, P: b.P.Value()}
	return dist.LogProb(b.value)
}

func (b *Binomial) Rand() float64 {
	dist := distuv.Binomial{N: b.N, P: b.P.Value(), Src: b.Src}
	return dist.Rand()
}

func (b *Binomial) Name() string {
	return b.name
}

func (b *Binomial) Value() float64 {
	return b.value
}

func (b *Binomial) SetValue(newValue float64) error {
	roundedVal := math.Round(newValue)
	if roundedVal < 0 || roundedVal > b.N {
		return &OutOfBoundsErr{"A binomial-distributed random variable can only take the integers between 0 and N as values."}
	}
	b.value = newValue

	return nil
}
