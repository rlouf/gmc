# GMC

GMC (Go Monte Carlo) is a probabilistic programming library written in Go. With
GMC you can perform Bayesian inference by defining a model using probability
distributions and sampling from the posterior of this model. GMC is intended to
be fast and fully tunable with reasonable defaults that make it easy to use.

## What is Bayesian inference?

Bayesian inference proceeds in two steps: defining a model with a set of
parameters that is supposed to generate the data, and computing the distribution
of the paramters conditioned on the data points. In some very specific examples,
it is possible to compute these distributions analytically and there is no need
for probabilistic programming; probabilistic programming is useful in the cases
where the posterior distribution has no simple analytical form.

### Define your model

Let's take a simple example: the coin toss experiment. We toss a coin N times,
and record the number of times the coin fell on head. We would like to know the
probability theta for the coin to fall on head. In GMC you would write:

```go
m := gmc.NewModel() // create a new model
theta := m.Beta('theta', 1, 1)
numHeads := m.Binomial('heads', N, theta)
```

Translated literaly:

> The number of heads after N tosses is given by a binomial distribution of 
> parameter theta. Before doing any experiment, we suppose that theta has a
> prior distribution Beta of parameters alpha = beta = 1

While the binomial is straightforward, it is not necessarily clear why theta
should have a Beta distribution. The reason is pragmatic: the distribution is
defined over [0,1] and  can take pretty much any shape when alpha et beta vary,
so it does not constraint the space of possible distribution too much.

### Prior predictive check

I picked alpha = beta = 1 but did not justify my choice. What if this
choice of hyperprior is not adapted to the problem? There is only one way to
know: create synthetic data from the model. If they sensibly match what we judge
to be the range of acceptable outcomes, then we are good. This step is called
the prior predictive check. Here's how to generate 1,000 of such samples with
GMC:

```go
samples := m.SamplePriorPredictive(1000)
```

### Sample from the posterior

Checked your prior? Good! Now you are ready to sample. GMC currently only packs
the Metropolis sampling algorithm.

```go
sampler := m.NewMetropolisHastings() // Initializes the sampler's configuration
trace :=  m.Sample(numSamples, sampler)
```

The output of the sampling is commonly called a trace. In GMC the trace is a 
`map[string][]float64` object that maps between the names of the variables and
the samples of their distribution.

### Post-sampling checks

Because you checked your prior and your sampler worked doesn't mean that the
inference is correct. There are two steps of checks:

#### Sampling check

The sampler can produce low-quality samples for a variety of reasons, so we need
to use a combination of diagnostics to make sure that is not the case. A first
good indicator is the effective sample size estimator, which roughly tells you
how many "truly independent" samples you got.

```go
ess := gmc.EffectiveSampleSize(trace)
```

#### Posterior check

As a final check, it is useful to see if the computed posterior is compatible
with your date. This is called the posterior predictive check.

```go
posterior := m.SamplePosteriorPredictive(trace)
```

## Licence

This library is distributed under the MIT licence. See the LICENCE.txt file in
the repository for more details.

## Contributions

Contributions are welcome. They can be as simple as submitting an issue if you
find a bug, advertising the library, and reaching out to me on Twitter.
