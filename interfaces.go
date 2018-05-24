package main

type Var interface {
	Value() float64
}

type RandVar interface {
	Name() string
	Value() float64
	SetValue(float64) error
	LogProb() float64
	Rand() float64
}
