# 🧠 Basic-Learning

**Basic-Learning** is a lightweight, zero-dependency deep learning framework coded entirely in pure Python. Designed for learners, tinkerers, and absolute control freaks who want to bend neural nets to their will—without downloading gigabytes of cruft.

No installations. No setup scripts. No excuses. Just raw, hackable Python. 🐍💥

---

## 📘 What is Basic-Learning?

Basic-Learning is your friendly neighborhood deep learning framework for those who want to *understand* what's really going on behind the scenes. No NumPy. No black-box abstractions. Every neuron, every gradient—at your command.

Whether you're crafting networks neuron-by-neuron or just want to peek inside backprop, this framework is your no-BS entry point into deep learning.

> ⚠️ Warning: It's basic on purpose. It's meant to teach you, not carry you.

---

## ⚙️ Core Features

### 🔌 Activation Functions (and their spicy derivatives):

* `sigmoid`, `swish`, `ReLU`, `tanh`, `leaky ReLU`

### 🔥 Loss Functions (yep, with derivatives too):

* `mean absolute error (MAE)`
* `mean squared error (MSE)`
* `binary cross entropy (BCE)` *(a.k.a. cel in the code)*

### 🧠 ForwardPass Class

* `nrn()` → One neuron
* `lyr()` → One layer
* `net()` → Whole network

### 🔁 Backpropagation (yes, it's custom)

* Supports `Neuron`, `Layer`, and `Network` levels
* Fine-grained control over learning

### 🧪 Initializer Class

* Comes with `constant`, `random`, `Xavier`, `He`, and `LeCun` strategies

### 🛠️ Model Class (a whole toolkit)

* `learn()` → Returns updated gradients for a single epoch
* `train()` → Same as `learn()`, but updates weights automatically
* `single()` → Train one sample over multiple epochs
* `multi()` → Overfit samples for science
* `iterative()` → Smart iterative training to avoid overfitting
* Supports full-batch training
* Save/load models with `pickle`

### 🧰 Utility Functions

* Transpose lists
* Compute elementwise means
* Clip values (gradient clipping is coming!)
* Developer-friendly debugging utilities

---

## 📦 How to Install

Clone it, crack it open, rule the neurons:

```bash
git clone https://github.com/fatal-error-404-samar/Basic-learning.git
cd Basic-learning
```

> No pip. No wheels. No setuptools sorcery. Just you and .py files.

---

## 🛣️ Roadmap

* ✅ Backpropagation and forward propagation are done
* 🔜 Mini-batch support
* 🔜 Optimizers (SGD is in, Adam is next)
* 🔜 Classification tools (softmax etc.)
* 🔜 Dropout and other regularization
* 🔜 NumPy version for faster ops (optional)
* 🔜 Plugin system for auto-magic high-level features

📆 **Major overhaul scheduled for July 2025** 🔧🧪

---

## 🙋‍♂️ About the Creator

Hey! I’m **Samar Jyoti Pator** — a student, a solo dev, a curious mind on a caffeine-fueled mission to demystify neural networks.

### 🚀 Experience

* Just 3 months of ML obsession (and this framework is proof!)
* Built from scratch while juggling life, code, and anime

### 🧠 Skills

* Python (with solid OOP chops)
* Core ML concepts (especially deep learning)
* Lua, C (dabbled in the low-level lands)
* A touch of NumPy (didn’t use it here out of choice)

### 🔗 My Other Repos

* [Rational](https://github.com/fatal-error-404-samar/Rational): Rational number arithmetic, precise and clean
* [A-Date-With-Luna](https://github.com/fatal-error-404-samar/A-date-with-luna): A heartfelt, terminal-based story game

Always building. Always learning. Always vibing. ✌️

---

## 🌐 Connect with Me

* 💬 **Discord**: [Join my server](https://discord.gg/q5WVtPSn)
* 📺 **YouTube**: [@thepacifist-hn3rr](https://www.youtube.com/@thepacifist-hn3rr)
* 📷 **Instagram**: [@samar\_loves\_anime](https://www.instagram.com/samar_loves_anime)
* 💼 **LinkedIn**: [Samar Jyoti Pator](https://www.linkedin.com/in/samar-jyoti-pator-8864aa367)
* 📧 **Email**: [samarjyotipator78@gmail.com](mailto:samarjyotipator78@gmail.com)

Reach out for collabs, convos, code reviews, or chaotic anime debates.

---

## 🪪 License

Licensed under the MIT License. TL;DR: Do whatever you want, just give credit where it's due and don't sue me if your model turns sentient.

---

**Built with Python, passion, and pure anime energy by Samar Jyoti Pator. 🐍✨**
