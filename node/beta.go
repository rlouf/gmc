package node

import (
	"fmt"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

type Beta struct {
	name  string
	value float64
	Alpha Var
	Beta  Var

	Src *rand.Rand
}

func NewBeta(name string, alpha, beta Var, src *rand.Rand) *Beta {
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
