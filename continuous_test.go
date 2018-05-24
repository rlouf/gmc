// Tests the continuous distributions.

package main

import (
	"testing"

	"golang.org/x/exp/rand"
)

func TestBetaSetValue(t *testing.T) {

	src := rand.New(rand.NewSource(1))
	alpha := Constant{1}
	beta := Constant{1}
	n := newBeta("test", alpha, beta, src)

	for i, tt := range []struct {
		value float64
		err   bool
	}{
		{1, false},
		{0, false},
		{-0.1, true},
		{1.1, true},
	} {
	}
}
