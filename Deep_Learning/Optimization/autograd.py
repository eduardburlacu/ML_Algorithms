"""
Write a Python class similar to the provided 'Value' class that implements the basic autograd operations:
    - addition
    - multiplication
    - ReLU activation
The class should handle scalar values and should correctly compute gradients
	for these operations through automatic differentiation.
"""

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(
            data= self.data + other.data,
            _children = (self, other),
            _op="+"
        )
        def _backward():
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(
            data= self.data * other.data,
            _children = (self, other),
            _op="*"
        )
        def _backward():
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.

    def relu(self):
        out = Value(
            data = max(0.0, self.data),
            _children=(self,),
            _op = "relu"
        )
        def _backward():
            self.grad = out.grad if self.data >= 0 else 0.
        out._backward = _backward
        return out

    def topological_sort(self):
        visited = set()
        sorted_vals = []
        def visit(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    visit(child)
                sorted_vals.append(node)
        visit(self)
        return sorted_vals

    def backward(self):
        self.grad = 1.
        for node in reversed(self.topological_sort()):
            node._backward()

if __name__ == "__main__":
    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    d = a * b
    e = c.relu()
    f = d.relu()
    loss = e + (-f); loss.backward()
    print(a) # Value(data=1.0, grad=0.0)
    print(b) # Value(data=2.0, grad=0.0)
    print(c) # Value(data=3.0, grad=1.0)
    print(d) # Value(data=2.0, grad=1.0)
    print(e) # Value(data=3.0, grad=1.0)
    print(f) # Value(data=2.0, grad=-1.0)
    print(loss) # Value(data=1.0, grad=1.0)
