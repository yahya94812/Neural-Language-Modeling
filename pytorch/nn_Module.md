# Inheriting from `nn.Module`

inheriting from `nn.Module` gives your class all the functionality needed to behave like a proper PyTorch neural network.

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
```

### Advantages of inheriting `nn.Module`

1. **Automatic parameter tracking**

   * Layers like `nn.Linear`, `nn.Conv2d`, etc. are automatically registered.
   * You can easily get all trainable parameters:

   ```python
   model.parameters()
   ```

2. **Built-in training and evaluation modes**

   ```python
   model.train()   # Enables training behavior
   model.eval()    # Disables Dropout, uses BatchNorm running stats
   ```

3. **Easy device movement**

   ```python
   model.to("cuda")
   model.to("cpu")
   ```

   All layers move together.

4. **Save and load models**

   ```python
   torch.save(model.state_dict(), "model.pth")
   model.load_state_dict(torch.load("model.pth"))
   ```

5. **Clean forward pass**
   You only define:

   ```python
   def forward(self, x):
       ...
   ```

   Then call the model like:

   ```python
   output = model(x)
   ```

   instead of `model.forward(x)`. PyTorch internally calls `forward()` and handles extra features like hooks.

### Without `nn.Module`

You would have to:

* Store parameters manually.
* Move tensors to GPU manually.
* Implement saving/loading yourself.
* Manage training/evaluation behavior yourself.
* Optimizers wouldn't know what parameters to update.

### One-line summary

> `nn.Module` is the **base class that turns your Python class into a PyTorch model**, automatically handling parameters, layers, GPU movement, saving/loading, and integration with optimizers and the training pipeline.



# Parameter Tracking Using `nn.Module`

The key idea is that **`nn.Module` overrides Python's attribute assignment** to automatically register layers and parameters.

### What happens behind the scenes?

When you write:

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
```

the line

```python
self.fc1 = nn.Linear(10, 20)
```

does **not** behave like a normal Python assignment.

`nn.Module` implements a special method called `__setattr__()`. Every time you assign an attribute (`self.something = value`), Python calls this method.

Internally, PyTorch does something conceptually like:

```python
def __setattr__(self, name, value):
    if isinstance(value, nn.Module):
        self._modules[name] = value      # Register child module
    elif isinstance(value, nn.Parameter):
        self._parameters[name] = value   # Register parameter
    else:
        object.__setattr__(self, name, value)
```

So when you assign:

```python
self.fc1 = nn.Linear(10, 20)
```

`fc1` is stored in a dictionary called `_modules`.

---

### Then what about weights?

`nn.Linear` is itself an `nn.Module`.

Inside `nn.Linear`'s constructor, it creates:

```python
self.weight = nn.Parameter(...)
self.bias = nn.Parameter(...)
```

Since `weight` and `bias` are `nn.Parameter` objects, they are automatically registered in the `_parameters` dictionary of that `Linear` module.

So you end up with a hierarchy like:

```
SimpleNet
│
├── _modules
│     ├── fc1  ---> Linear
│     └── fc2  ---> Linear
│
└── parameters()
       │
       ├── fc1.weight
       ├── fc1.bias
       ├── fc2.weight
       └── fc2.bias
```

---

### How does `model.parameters()` work?

It recursively walks through every registered child module:

```python
model
 ├── fc1
 │     ├── weight
 │     └── bias
 └── fc2
       ├── weight
       └── bias
```

and yields all the `nn.Parameter` objects it finds.

That's why this works:

```python
for p in model.parameters():
    print(p.shape)
```

Output:

```
torch.Size([20, 10])
torch.Size([20])
torch.Size([5, 20])
torch.Size([5])
```

---

### Why use `nn.Parameter`?

A plain tensor is **not** considered a trainable parameter:

```python
self.x = torch.randn(10, 10)      # Not registered
```

But if you wrap it:

```python
self.x = nn.Parameter(torch.randn(10, 10))
```

it is automatically registered, appears in `model.parameters()`, and the optimizer will update it during training.

### In one sentence

`nn.Module` uses Python's special `__setattr__()` method to intercept assignments like `self.layer = ...`; it registers `nn.Module` objects as child modules and `nn.Parameter` objects as trainable parameters, allowing methods like `model.parameters()` to recursively discover everything that should be trained.
