package main

import (
	"log"
	"math"
)

type Constant struct {
	value float64
}

func (c *Constant) Value() float64 {
	return c.value
}

// The SumGate represents the sum of two variables.
// Its value is equal to the sum of the values of the variables.
type SumGate struct {
	X Var
	Y Var
}

func (s SumGate) Value() float64 {
	return s.X.Value() + s.Y.Value()
}

// The ProdGate represents the product of two variables.
// Its value is equal to the product of the values of the variables.
type ProdGate struct {
	X Var
	Y Var
}

func (p ProdGate) Value() float64 {
	return p.X.Value() * p.Y.Value()
}

// The Logistic gate applies the logistic function to a variable.
//
// If we note x the value of the variable X, the value of the logistic gate is
// given by:
//
// 1 / (1 + exp(-x))
//
// For more info see: https://en.wikipedia.org/wiki/Logistic_function
type LogisticGate struct {
	X Var
}

func (l *LogisticGate) Value() float64 {
	v := l.X.Value()
	if v >= 0 {
		z := math.Exp(-v)
		return 1 / (1 + z)
	}
	z := math.Exp(v)
	return z / (1 + z)
}

// The LogitGate applies the logit (or log-odds) function to a variable.
// If we note x the value of the variable X, the value of the LogitGate is
//
// log(1 / (1 - x))
//
// The logit function is only defined for x in [0,1] and the function will
// panic if x is out of bounds.
//
// For more info see: https://en.wikipedia.org/wiki/Logit
type LogitGate struct {
	X Var
}

func (l *LogitGate) Value() float64 {
	v := l.X.Value()
	if v < 0 || v > 1 {
		log.Panicf("logit function is defined on [0,1], got %f", v)
	}

	return math.Log(1 / (1 - v))
}

// The SwitchGate chooses between the values of two variables depending on the
// value of a third variable and a threshold.
//
// Let us note L (l) and  R (r) the two variables (respectively their values),
// t the (real) value of the threshold, S the switch variable with value s. Then:
// 	  if s <= t: the gate's value is l
//    if s > t : the gate's value is r
type SwitchGate struct {
	threshold float64
	Switch    Var
	Left      Var
	Right     Var
}

func (s *SwitchGate) Value() float64 {
	t := s.Switch.Value()
	if t <= s.threshold {
		return s.Left.Value()
	}
	return s.Right.Value()
}
