# Long-Shot Ideas

Papers and directions that are months away but worth tracking for future bridging.

---

## On the Geometry of Recurrent Spiking Networks (Casco-Rodriguez, 2025)

**Source:** OpenReview UOAyw2m5r1 (under review, Symmetry & Geometry in Neural Representations workshop)
**Author:** Josue Casco-Rodriguez, Rice University

**Core idea:** Theory of SNNs as piecewise-*constant* functions (vs piecewise-linear for ReLU nets). At each timestep, each spiking neuron (LIF: u_t = λ⊙(1−s_{t−1})⊙u_{t−1} + W_x·x_t + W_r·s_{t−1}, s_t = Heaviside(Re(u_t)−ϑ)) partitions the unrolled input space ℝ^{T×d} into spike/no-spike regions. Algorithm: SplineCam-SNN for exact region computation.

**Key findings:**
1. Hidden layers need **skip connections** to draw new partitions (unlike continuous nets where each layer creates new boundaries from previous ones — in SNNs, previous outputs are binary/constant within regions, so hidden layers can only label existing regions without skip connections)
2. **Leakages, delays, input weights** rotate partition boundaries; **recurrent weights** can only *shift* them (binary signals → limited geometric influence)
3. Gradient descent aligns partitions around data; STDP (local bio-plausible learning) can only shift boundaries based on data density, not rotate → STDP is geometrically limited
4. Implication: future SNN plasticity algorithms should train leakages and delays, not just synaptic weights

**Bridge to our work:**
- Piecewise-constant (SNN) vs piecewise-linear (ReLU): different function class, same geometric framework of input-space partitions, boundaries, and regions
- If we can compress ReLU nets by merging equivalent linear regions, the same idea maps to merging equivalent constant regions in SNNs — but simpler (constant = just a spike pattern label, no affine map to preserve)
- The skip-connection finding is interesting: it means SNN depth without skip connections doesn't add new partition complexity, only relabels. This is the opposite of our ReLU setting where depth multiplies region count exponentially
- SNN compression could be much more aggressive than ReLU compression because the function within each region is trivial (binary spike pattern). The geometry is entirely in the partition boundaries.
- Neuromorphic hardware (Loihi, SpiNNaker) is where compression matters most — power budgets are 10-1000× tighter than GPU

---

## Hippoformer: Integrating Hippocampus-inspired Spatial Memory with Transformers (2025)

**Source:** OpenReview hxwV5EubAw (under review, ICLR 2025); [lucidrains implementation](https://github.com/lucidrains/hippoformer)
**Authors:** Anonymous (blind review)

**Core idea:** Hybrid architecture combining a hippocampal-inspired structural memory (mm-TEM, based on meta-MLP relational memory) with a standard Transformer. The two modules run in parallel: the Transformer provides precise short-term key-value memory, while mm-TEM provides structured but lossy long-term spatial memory. Together they enable spatial reasoning that neither component achieves alone.

**Key details:**
1. mm-TEM forms **grid-like representations** (analogous to entorhinal grid cells) — prediction horizon controls grid scale
2. MEC provides structural codes, hippocampus binds them with sensory codes → flexible spatial inference
3. Outperforms pure Transformers and pure TEM on 2D and 3D grid prediction/generalization tasks
4. The dual-memory architecture (precise short-term + structured long-term) mirrors hippocampal-neocortical complementary learning systems

**Bridge to our work:**
- Grid cells = periodic spatial basis functions. The mm-TEM learns a structured low-dimensional representation of space — essentially a learned geometric embedding. If we're compressing networks that process spatial/sequential data, the grid-cell representation could be a natural compressed basis
- The complementary memory idea (precise but expensive vs structured but cheap) maps onto our correction framework: the quantized network is the "cheap structured memory," the correction is the "precise overlay"
- Long-term: if geometric compression can be framed as learning a structured spatial memory over the weight manifold (weights as positions in a space, compression as efficient spatial encoding), the hippocampal analogy might give us architectural priors for the correction network
