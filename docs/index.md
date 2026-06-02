<section class="neat-hero">
  <div>
    <p class="neat-eyebrow">Keras-first optimizer library</p>
    <h1 class="neat-title">Nash-Equilibrium Adaptive Training</h1>
    <p class="neat-lede">
      NEAT adds conflict-aware gradient correction to neural network training.
      Use it as a Keras 3 optimizer, validate the update rule in NumPy, and
      inspect training diagnostics while experiments run.
    </p>
    <div class="neat-actions">
      <a class="neat-button neat-button-primary" href="quickstart/">Quickstart</a>
      <a class="neat-button" href="api/">API Reference</a>
      <a class="neat-button" href="research/math-spec/">Math Spec</a>
    </div>
  </div>
  <div class="neat-panel">

```python
import keras
from neat_optim import NEAT

model.compile(
    optimizer=NEAT(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
```

  </div>
</section>

<section class="neat-grid">
  <div class="neat-card">
    <h3>Keras Optimizer</h3>
    <p>Drop NEAT into <code>model.compile(...)</code> for standard Keras training workflows.</p>
  </div>
  <div class="neat-card">
    <h3>Reference Engine</h3>
    <p>Use the NumPy implementation for deterministic validation and algorithm research.</p>
  </div>
  <div class="neat-card">
    <h3>Player-Aware Mode</h3>
    <p>Treat examples or tasks as gradient players in custom TensorFlow loops.</p>
  </div>
</section>

## Install

```bash
pip install "neat-optim[keras]" tensorflow
```

Core engine only:

```bash
pip install neat-optim
```

## Docs

- [Quickstart](quickstart.md): installation and first training examples
- [API](api.md): public objects, configuration, and diagnostics
- [Research](research/math-spec.md): update equations and algorithm notes
- [Contributing](contributing/development.md): development and release workflow
