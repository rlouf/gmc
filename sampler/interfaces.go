package sampler

type Sampler interface {
	Sample()
	Tune()
	Initialize()
}
