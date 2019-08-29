package node

type OutOfBoundsErr struct {
	msg string
}

func (o *OutOfBoundsErr) Error() string {
	return o.msg
}
