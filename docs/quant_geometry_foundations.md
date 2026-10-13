# Quantization geometry: deformation, repair, and an exact ledger

Quantizing a layer's weights changes the function the layer computes. This note asks: **how much of that change can a linear map applied to the quantized layer's output undo, and why does the rest remain?** The answer is a ledger. The correction needed to get back to full precision (FP) splits exactly into three terms.

**Contents.** [The ledger at a glance](#the-ledger-at-a-glance) ·
Part I, a square layer: [deformation](#1-how-quantization-deforms-a-layers-output), [linear repair](#2-what-a-linear-map-on-the-quantized-output-can-repair), [ReLU](#3-relu-repairing-before-or-after-it-gives-different-results), [ledger](#4-the-ledger-for-the-square-layer) ·
Part II, fewer outputs than inputs: [fibers and the residual](#5-switch-to-a-layer-with-fewer-outputs-than-inputs), [complete ledger](#6-the-complete-ledger), [cancellation](#7-the-terms-can-cancel), [networks and the oracle](#8-from-one-layer-to-a-network), [calibration fitting](#9-comparison-fitting-a-correction-to-calibration-data) ·
[More examples of $U$](#a-more-examples-of-the-residual-u) · [Technical details](#b-technical-details).
An animated version of Figures 1–2 can be generated as described in the [implementation README](quant_geometry_implementation.md#reproduce-the-artifacts).

## The ledger at a glance

Take one layer with input row vector $x$, FP weights $W$, quantized weights $W_q$ and ReLU $\phi$.

- The FP layer outputs $y=\phi(xW)$. The quantized layer outputs $h=\phi(z_q)$, where $z_q=xW_q$ is the quantized preactivation.
- $R$ is the best **linear** repair: the matrix that brings $W_qR$ as close as possible to $W$, $R=\operatorname{argmin}_R\lVert W-W_qR\rVert_F$. It is computed from the weights alone, with no data, and kept at full precision.
- $U=W-W_qR$ is the part of $W$ that no linear map applied after $W_q$ can reproduce.

Geometrically, a layer with fewer outputs than inputs flattens its input space: inputs that differ only along certain directions give the same output. This already happens at full precision. Quantization **tilts the flattened directions**. Inputs that $W_q$ sends to the same point, but $W$ does not, cannot be separated again by anything applied after $W_q$. $U$ is what $W$ does along those directions. $U=0$ exactly when $W_q$ flattens no direction that $W$ uses. This holds for square layers whose $W_q$ stays invertible, and for layers with more outputs than inputs and full rank. It generally fails for layers with fewer outputs than inputs, such as convolutions and MLP down-projections.

The correction needed, $y-h$, splits into three terms:

| Term | Name | Definition | In words |
|---|---|---|---|
| $c_R$ | map repair | $hR-h$ | The correction $R$ makes when applied to the quantized output $h$. |
| $c_{\rm NL}$ | activation-order error | $\phi(z_qR)-hR$ | How much better $R$ would have done before ReLU. ReLU has already set some coordinates of $h$ to zero, and $R$ cannot recover them. |
| $c_U$ | residual deformation | $y-\phi(z_qR)$ | What $R$ misses even before ReLU. $W_q$ flattens some input directions that $W$ does not; inputs that differ along them have the same quantized preactivation but different FP outputs. |

$$
y-h=c_R+c_{\rm NL}+c_U.
$$

Each term needs more information to correct than the one before. $c_R$ needs only the layer output $h$, $c_{\rm NL}$ needs the preactivation $z_q$, and $c_U$ needs the original input $x$.

**Reading the figures.** Every figure uses the same key, and each repeats it in a legend:

| What you see | Meaning |
|---|---|
| orange, dashed, hollow markers | the FP target |
| gray, solid | the quantized layer, uncorrected |
| blue, solid | after a correction |
| near-black | inputs, before any map |
| blue / green / purple arrows | $c_R$ / $c_{\rm NL}$ / $c_U$ |
| black dashed arrow | $c_{\rm total}$ |

Marker shape tells inputs apart: circles for $A$ and $C$, squares for $B$ and $D$. Where $C$ and $D$ appear together they need different FP targets, so $D$'s target is purple instead of orange, and an arrow that belongs to only one of them is drawn in that point's color.

**Two example layers.** Part I uses a square layer $W_1$ whose quantized weights stay invertible. Its map repair is exact ($U=0$), so ReLU is the only source of error. Part II switches once, to a layer $W_2$ with three inputs and two outputs. There quantization tilts the flattened direction, the repair is incomplete, and $c_U$ appears. Part II keeps $W_2$ to the end, including the network section and the comparison with calibration fitting. The appendix uses other matrices for a zoomed-in view and for an atypical case, and says so each time.

Inputs are row vectors $x\in\mathbb R^{1\times n}$, and $W,W_q\in\mathbb R^{n\times m}$. The main text uses zero bias; bias is handled in [Technical details](#b3-bias-and-the-sign-convention-in-the-code).

# Part I — a square layer: the repair is exact, but ReLU gets in the way

## 1. How quantization deforms a layer's output

Start with the linear part of the layer. FP sends $x$ to $xW$; the quantized layer, Q, sends it to $xW_q$. Throughout Part I,

$$
W_1=\begin{bmatrix}1&-0.4\\0.65&0.85\end{bmatrix},\qquad
W_{1,q}=\begin{bmatrix}1&0\\1&1\end{bmatrix},
$$

where 2-bit symmetric quantization rounds each coefficient to $\{-1,0,1\}$. In Part I, $W$ and $W_q$ mean these two matrices.

![The same input shape under FP, quantized, and repaired preactivation maps](figures/quant_geometry/geometry_deformation.png)

**Figure 1 — the quantized layer deforms the output, and a linear map can undo it.** A grid, a circle and two inputs $A=(-0.8,-0.1)$ and $B=(-0.6,-0.2)$ pass through each map. FP and Q produce different shapes. Applying $R$ to the Q output recovers the FP output exactly: the two right panels arrive at the same points, one by $xW$ and the other by $(xW_q)R$. There is no activation yet. The axes are the same in every panel, so the change in shape comes from the map, not the viewpoint.

## 2. What a linear map on the quantized output can repair

$R$ can mix and rescale the output coordinates, moving the whole Q output back toward FP. **$R$ is kept at full precision; it is not rounded to the quantization grid.** We choose it by

$$
R\in\underset{R\in\mathbb R^{m\times m}}{\operatorname{argmin}}\;\|W-W_qR\|_F^2.
$$

This objective weights every input direction equally and uses no data.

When $W_q$ is square and invertible, the repair is exact. No input direction has been lost, so we can undo the quantized map and apply the FP map instead:

$$
R=W_q^{-1}W,\qquad (xW_q)R=xW.
$$

<!-- For $W_1$, $R=\left[\begin{smallmatrix}1&-0.4\\-0.35&1.25\end{smallmatrix}\right]$. -->

## 3. ReLU: repairing before or after it gives different results

The layer applies ReLU to the preactivation. ReLU sets every negative coordinate to zero, so a shape gets pressed flat against the axes and different points can land on the same point. Formally, $\phi(v)_j=\max(v_j,0)$, and

$$
z_q=xW_q,\qquad h=\phi(z_q),\qquad y=\phi(xW).
$$

Because $R$ repairs $W_1$ exactly, applying it before ReLU gives the FP output: $y=\phi(z_qR)$. A correction applied to the layer's output, however, only sees the output after ReLU. Applying a map and then ReLU generally differs from applying ReLU and then the map:

$$
\phi(z_q)R\ne\phi(z_qR).
$$

![An exact preactivation repair before ReLU versus a failed repair after ReLU](figures/quant_geometry/geometry_relu.png)

**Figure 2 — repair then ReLU, versus ReLU then repair.** Same $W_1$, $R$, grid, circle, $A$ and $B$ as Figure 1, now with ReLU. Repairing first gives $A$ and $B$ different FP outputs, $(0,0.235)$ and $(0,0.07)$. Applying ReLU first sends both to zero, so $R$ receives the same point for both and cannot separate them. The linear map was fully repairable; the problem comes from ReLU.

## 4. The ledger for the square layer

For input $x$, the quantized layer outputs $h$ and the FP layer outputs $y$. The correction needed is

$$
c_{\rm total}=y-h.
$$

Applying $R$ to the output moves $h$ to $hR$. That step is the map repair:

$$
c_R=hR-h,\qquad c_{\rm total}=c_R+(y-hR).
$$

![The correction needed, the correction made by R, and what is still needed](figures/quant_geometry/ledger_geometry_uncomp.png)

**Figure 3 — correction needed, correction made by $R$, correction still needed.** Same $W_1$ and inputs as Figures 1–2. $R$ moves the shape, but $A$ and $B$ start at zero and stay there, so they still need their full correction. Because $R$ acts after ReLU, some points of $hR$ have negative coordinates.

For $W_1$ the repair is exact, so everything $R$ leaves uncorrected is due to applying it after ReLU:

$$
c_{\rm NL}=\phi(z_qR)-hR=y-hR,\qquad c_{\rm total}=c_R+c_{\rm NL}.
$$

![The activation-order error points from hR toward the exact FP target](figures/quant_geometry/ledger_geometry_nonlinear.png)

**Figure 4 — $hR\to\phi(z_qR)=y$.** The correction still needed in Figure 3 is exactly the difference between repairing after ReLU and repairing before it (Figure 2). Applying ReLU first leaves $A$ and $B$ together at zero; the arrows take them to their different FP outputs.

This ledger has only two terms because $W_1$ was chosen so that $U=0$. Part II drops that assumption.

# Part II — a layer with fewer outputs than inputs: the repair is incomplete

## 5. Switch to a layer with fewer outputs than inputs

Square layers whose quantized weights stay invertible are common, for example the attention projections in a transformer. But many layers have fewer outputs than inputs. A convolution maps a patch to output channels: for a $3\times3$ convolution with 64 input and 64 output channels, each output location reads $64\cdot3\cdot3=576$ input values and produces 64. In row-vector form that is a $576\times64$ map, which flattens 512 input directions even at full precision. An MLP down-projection similarly maps a wide hidden layer back to the model width.

From here to the end of the main text we use a small layer of the same kind, with three inputs and two outputs:

$$
W_2=\begin{bmatrix}1&0.25\\0.75&1\\0.25&0.8\end{bmatrix},\qquad
W_{2,q}=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix}.
$$

Again 2-bit quantization rounds to $\{-1,0,1\}$, and $W$ and $W_q$ now mean these two matrices. Both have rank 2, the maximum possible; the singular values of $W_q$ are $\sqrt3$ and $1$, so no direction is close to being lost.

**The repair from Part I now only goes part of the way.** Each map keeps two of the three input dimensions and flattens the third: moving an input along the flattened direction does not change the output. $W_q$ flattens $d=(1,-1,1)$, since $dW_q=0$. $W$ flattens a different direction, because quantization has tilted it; in particular $dW=(0.5,0.05)\ne0$.

**Fibers.** The set of all inputs that a map sends to the same output is called a *fiber* of the map. For a layer with three inputs and two outputs, each fiber is a line parallel to the flattened direction, and the fibers of $W_q$ are the lines parallel to $d$. Every input on one fiber of $W_q$ has the same quantized output, so nothing computed from that output can tell them apart. Because quantization tilted the flattened direction, those inputs lie on *different* fibers of $W$, so their FP outputs differ. That is exactly the part of the error a correction applied to the output cannot fix.

$R$ is still the least-squares solution from Section 2. Let $P$ be the projection onto the input directions $W_q$ keeps. Then (derived in [B.1](#b1-constructing-the-map-repair))

$$
W_qR=PW,\qquad U=W-W_qR=(I-P)W,\qquad xW=(xW_q)R+xU.
$$

**$R$ reproduces what $W$ does on the input directions $W_q$ keeps. $U$ is what $W$ does on the direction $W_q$ flattens.** For $W_2$,

$$
R=\begin{bmatrix}0.8\overline3&0.2\overline3\\0.08\overline3&0.78\overline3\end{bmatrix},\qquad
U=\tfrac16\begin{bmatrix}1&0.1\\-1&-0.1\\1&0.1\end{bmatrix}.
$$

$xU$ depends on the original input, so no correction applied to the layer's output can produce it. The ledger uses it to account for the error, not as a step of the repair.

![Two inputs on the same fiber of the quantized map get the same quantized output but different FP outputs](figures/quant_geometry/geometry_rectangular_fibers.png)

**Figure 5 — two inputs the quantized layer cannot tell apart.** 1: in input space, each map flattens one direction; quantization has tilted it. $C=(0.8,-0.2,-0.1)$ and $D=(0.2,0.4,-0.7)$ differ only along $d$ ($C-D=0.6\,d$), so they lie on the same fiber of $W_q$ (gray line). 2: they get the same quantized output. 3: their FP outputs differ. 4: $R$ moves the shared point to the best common position, but it is still one point; the remaining arrows are the different $xU$. There is no activation in this figure.

Any correction that only sees the quantized output of $C$ and $D$ must give both the same answer, but FP gives them different outputs. This is a limit on the information available, not on the repair method. The general condition is in [B.2](#b2-when-the-repair-is-exact), and appendix A gives two more examples.

## 6. The complete ledger

With $U\ne0$, repairing before ReLU no longer reaches FP: in general $y=\phi(z_qR+xU)\ne\phi(z_qR)$. What $R$ leaves uncorrected now has two parts, one for each limitation of the repair:

$$
y-hR=
\underbrace{\phi(z_qR)-hR}_{c_{\rm NL}:\ \text{ReLU acted before the repair}}
+\underbrace{y-\phi(z_qR)}_{c_U:\ \text{the linear repair is incomplete}},
\qquad c_U=\phi(z_qR+xU)-\phi(z_qR).
$$

![The residual deformation points from the repaired output toward FP](figures/quant_geometry/ledger_geometry_range.png)

**Figure 6 — $\phi(z_qR)\to y$.** Two copies of the grid and circle, one shifted along $d$ to pass through $C$ and one to pass through $D$. Shifting along $d$ moves each input along its fiber of $W_q$, so it does not change the quantized output. Left: both copies give the same repaired output. Middle: their FP targets differ (orange for $C$, purple for $D$). Right (zoomed): $C$ and $D$ start from the same point and need opposite corrections, $c_U=(0.15,0)$ and $(-0.15,0)$.

The full ledger follows one path,

$$
h\ \longrightarrow\ hR\ \longrightarrow\ \phi(z_qR)\ \longrightarrow\ y,
\qquad
\boxed{c_{\rm total}=c_R+c_{\rm NL}+c_U.}
$$

The terms add up to $y-h$ by construction: each intermediate point is added in one term and subtracted in the next. What carries meaning is the choice of intermediate points. Each one is the best correction possible with more information: first $h$, then $z_q$, then $x$. A different path, or a different $R$ where $R$ is not unique, would split the same total differently ([B.4](#b4-the-split-depends-on-the-choice-of-r)). We always use this path and the $R$ constructed in B.1.

## 7. The terms can cancel

The terms are vectors, so a later term can undo an earlier one. Their lengths do not add up to the length of the total correction.

![C and D share every step up to the residual, then diverge: for C the three terms nearly cancel, for D they reinforce](figures/quant_geometry/ledger_geometry_cancellation.png)

**Figure 7 — the full ledger for $C$ and $D$.** $C$ and $D$ have the same $h=(0.6,0)$, so they have the same $c_R=(-0.1,0.14)$ and $c_{\rm NL}=(-0.025,-0.14)$: $R$ moves the point off the axis, and the activation-order error moves it back. Only $c_U$ differs. For $C$, the three terms have lengths $0.17$, $0.14$ and $0.15$, but the total is only $(0.025,0)$. For $D$, the horizontal parts of all three terms point the same way, and the total is $(-0.275,0)$.

To compare the terms over a dataset, use their inner products ([B.5](#b5-interaction-between-terms-the-gram-identity)), not just their lengths.

## 8. From one layer to a network

In a network, the input $a$ reaching a layer may already differ from its FP value $a_{\rm fp}$. Compare three outputs: the quantized output $h_q=\phi(aW_q)$, the FP layer's output on the same input, $g=\phi(aW)$, and the FP network's output, $y_{\rm ref}=\phi(a_{\rm fp}W)$. This separates the error this layer adds from the error it received:

$$
y_{\rm ref}-h_q=
\underbrace{g-h_q}_{c_{\rm local}}+
\underbrace{y_{\rm ref}-g}_{c_{\rm incoming}}.
$$

$c_{\rm local}$ is the Section 6 ledger applied to the input the layer actually receives: $c_{\rm local}=c_R(a)+c_{\rm NL}(a)+c_U(a)$. $c_{\rm incoming}$ is the earlier error after passing through this layer's FP map.

**The oracle** adds $y_{\rm ref}-h_q$ to a layer's output, so the next layer receives its FP input. Applying it after every layer isolates the error each layer creates on its own. Applying it only after a later layer removes all error accumulated up to that point; that tells you about a particular layer only if it was the only source of error in between.

![Incoming input error, local error, incoming error at the output, and the oracle correction](figures/quant_geometry/repair_network_balance.png)

**Figure 8 — error added by this layer versus error it received, on $W_2$.** The FP input is $a_{\rm fp}=D$; the input actually received is $a=(0.4,0.4,-0.4)$. Right: the local part goes from $h_q=(0.8,0)$ to $g=(0.6,0.18)$, and the incoming part from $g$ to $y_{\rm ref}=(0.325,0)$. The oracle applies the whole correction $y_{\rm ref}-h_q$ in one step.

[Implementation](quant_geometry_implementation.md) and [next experiments](quant_geometry_progress.md) are documented separately.

## 9. Comparison: fitting a correction to calibration data

The ledger chooses $R$ from the weights alone. The usual alternative fits a correction to the errors observed on a calibration set $\mathcal D$. With the same full-precision linear correction and the same layer $W_2$, two such fits are

$$
R_{\rm pre,data}\in\operatorname{argmin}_R\sum_{x\in\mathcal D}\|xW_qR-xW\|^2,\qquad
R_{\rm post,data}\in\operatorname{argmin}_R\sum_{x\in\mathcal D}\|\phi(xW_q)R-\phi(xW)\|^2.
$$

| Correction | Objective | What it uses |
|---|---|---|
| $R$ (map repair) | $\lVert W-W_qR\rVert_F^2$ | the weights alone; every input direction weighted equally |
| $R_{\rm pre,data}$ | preactivation error on $\mathcal D$ | the input distribution and its correlations |
| $R_{\rm post,data}$ | output error on $\mathcal D$ | the outputs after ReLU, and whatever correlations survive ReLU |

All three are least-squares fits, but they answer different questions. The data fits find the matrix that best reproduces FP outputs on this data. They do not say which part of the quantization error that matrix removes, and their loss cannot tell you.

![Four columns compare Q, map repair, preactivation data fit, and output data fit, before and after ReLU](figures/quant_geometry/geometry_repair_vs_reconstruction.png)

**Figure 9 — three objectives give three different matrices.** The columns show Q with no correction, then the three fitted matrices. The top row applies each matrix to the quantized preactivation; the bottom row applies it to the quantized output after ReLU. The calibration inputs are correlated with the flattened direction $d$ (construction in [B.6](#b6-calibration-set-for-figure-9)), so on this data the FP output along $d$ can be partly predicted from the quantized output. The grid and circle show a subset of the inputs; the fits use the full calibration set.

On this example:

| | Map error $\lVert W-W_qR\rVert_F^2$ | Calibration output MSE |
|---|---|---|
| Q, no correction | 0.228 | 0.0236 |
| $R$ (map repair) | **0.084** | 0.0257 |
| $R_{\rm pre,data}$ | 0.332 | 0.0140 |
| $R_{\rm post,data}$ | 0.361 | **0.0101** |

The output fit has the lowest calibration error, yet its corrected map $W_qR_{\rm post,data}$ is **further from $W$ than $W_q$ is**. It does not undo the quantization error. It finds a different matrix, which uses correlations in this calibration set to predict the FP output, including the part along the direction $W_q$ flattens. That prediction only holds as long as those correlations do. The map repair is the reverse: it is the closest to $W$, but on this calibration set it does slightly worse than no correction, both before ReLU (MSE $0.0497$ versus $0.0483$) and after it. The calibration inputs are concentrated along the flattened direction, and the map objective gives that direction no extra weight.

This is a single constructed example. It shows that the two kinds of fit can disagree sharply, not how often they do in trained networks. The ledger makes that question measurable: for any correction, compute which of $c_R$, $c_{\rm NL}$ and $c_U$ it reduces.

## A. More examples of the residual $U$

These examples use matrices other than $W_2$, for the reasons given in each.

### A.1 Zoomed in: one output, two inputs

To draw fibers as lines in the plane, look at **a single output of a layer with two inputs**. This is the smallest case of the effect in Section 5, not a new example. In this subsection only, $x$ and the weight vector $w$ are **column vectors**. The output is the number $x^\top w$. The direction of $w$ is the input direction the output responds to, and its length is the gain.

Quantization changes $w$ into $w_q$, which in general has a different length and direction. Suppose we keep only the quantized output and try to repair it by multiplying by a number $r$. Changing $r$ stretches or flips $w_q$ along its own line, but cannot rotate it toward $w$. So an exact repair for every input exists only when $w$ lies on that line:

$$
r\,x^\top w_q=x^\top w\quad\text{for every }x
\quad\Longleftrightarrow\quad w=r w_q.
$$

![Weight-vector projection and rotated fibers; two inputs have equal quantized outputs but different FP outputs](figures/quant_geometry/geometry_scalar_fibers.png)

**Figure A1 — the weight rotates without losing rank.** Left: $w$ and $w_q$ differ in length and direction. The blue arrow $rw_q$ is the projection of $w$ onto the line through $w_q$, which is as far as rescaling can go. The purple arrow $u$ is the rest. Right: each line is a fiber, a set of inputs with the same output. $A$ and $B$ lie on the same quantized fiber (gray) but on different FP fibers (dashed orange).

2-bit quantization changes $w=(1,0.55)^\top$ to $w_q=(1,1)^\top$. Both have rank 1, as any nonzero single-output map must. **Quantization has rotated the fibers without reducing rank.** The best gain and the remainder are

$$
r=\frac{w_q^\top w}{w_q^\top w_q},\qquad
w=r w_q+u,\qquad w_q^\top u=0,\qquad
x^\top w=r\,(x^\top w_q)+x^\top u.
$$

The first term can be computed from the quantized output alone. The second depends on the component of the input along $u$, which is perpendicular to $w_q$ and so invisible to the quantized output. Here each fiber of the quantized output, $F_q(c)=\{x:x^\top w_q=c\}$, is a line perpendicular to $w_q$. Multiplying by $r$ changes the value assigned to each fiber but not which inputs share one. If $u\ne0$, moving an input along $u$ keeps it on the same quantized fiber but changes its FP output, since $u^\top w_q=0$ while $u^\top w=\|u\|^2>0$.

For $A=(0.8,-0.2)^\top$ and $B=(-0.2,0.8)^\top$: $A^\top w_q=B^\top w_q=0.6$, while $A^\top w=0.69$ and $B^\top w=0.24$. The projection gives $r=0.775$ and $u=(0.225,-0.225)^\top$. The repair outputs $0.465$ for both, so $A$ still needs $+0.225$ and $B$ needs $-0.225$.

### A.2 Atypical: quantization that reduces rank

Sections 5–7 showed information loss without any loss of rank. Less commonly, quantization can also reduce a map's rank, for example when rounding sets a whole row to zero. To show this case we construct a separate matrix $W_3$:

$$
W_3=\begin{bmatrix}1&0.3\\0.25&0.25\end{bmatrix},\qquad
W_{3,q}=\begin{bmatrix}1&0\\0&0\end{bmatrix},\qquad
R=\begin{bmatrix}1&0.3\\0&1\end{bmatrix},\qquad
U=\begin{bmatrix}0&0\\0.25&0.25\end{bmatrix}.
$$

![A collapsed square, a repair of the line it collapses to, and input-dependent corrections restoring its width](figures/quant_geometry/geometry_loss.png)

**Figure A2 — quantization collapses a dimension.** Panel 1: the FP outputs of $A=(0.4,-0.8)$ and $B=(0.4,0.8)$ differ. Panel 2: quantization sends both to $(0.4,0)$, and the whole square collapses onto a line. Panel 3: $R$ moves their shared output to the best common point $(0.4,0.12)$, but cannot separate them or restore the square's width. Panel 4: adding their different $xU$, $(-0.2,-0.2)$ for $A$ and $(0.2,0.2)$ for $B$, recovers the FP output. There is no activation in this figure.

Here the second input direction is lost entirely. The mechanism is the same as in Section 5, where a direction is lost only because the layer has fewer outputs than inputs.

## B. Technical details

### B.1 Constructing the map repair

Among all minimizers, we choose the one closest to leaving the output unchanged. Write $R=I_m+E$ and take the minimum-norm least-squares solution:

$$
\min_E\|W-W_q-W_qE\|_F^2,\qquad E=W_q^\dagger(W-W_q).
$$

With the exact Moore–Penrose pseudoinverse,

$$
R=I_m+W_q^\dagger(W-W_q),\qquad
U=(I_n-W_qW_q^\dagger)(W-W_q).
$$

$R$ is $m\times m$ and $U$ is $n\times m$. With $P=W_qW_q^\dagger$, $W_qR=PW$ and $U=(I_n-P)W$. $P$ projects onto the input directions $W_q$ keeps, and $W_q^\top U=0$. This orthogonality is between the maps; it does not make the output corrections perpendicular for each input.

The best repaired map $W_qR$ is unique, but $R$ itself is not unique when $W_q$ has rank below $m$; choosing the solution closest to the identity fixes it. The map objective also equals, up to a constant factor, the mean squared preactivation error for inputs drawn uniformly from the unit sphere.

For the preactivation data objective, let $\Delta_R=W-W_qR$ and $N=|\mathcal D|$:

$$
S_{\mathcal D}=\frac1N\sum_{x\in\mathcal D}x^\top x,\qquad
\frac1N\sum_{x\in\mathcal D}\|x\Delta_R\|^2
=\operatorname{tr}(\Delta_R^\top S_{\mathcal D}\Delta_R).
$$

$S_{\mathcal D}$ is the uncentered second moment of the inputs. If $S_{\mathcal D}=cI_n$ with $c>0$, this objective is a multiple of the map objective and has the same minimizers. Any other $S_{\mathcal D}$ weights map error more in some directions than others; this need not change the minimizer when an exact repair exists. The objective after ReLU generally cannot be written this way, because both the corrected output and the target have passed through ReLU.

### B.2 When the repair is exact

An invertible square $W_q$ is enough for an exact repair, but not necessary. If $W_q$ has full row rank (possible only when $m\ge n$), $W_qW_q^\dagger=I_n$, so $x$ can still be recovered from $xW_q$ and $U=0$. What matters is **which input directions $W_q$ keeps**, not whether it is square.

For arbitrary inputs and the exact pseudoinverse,

$$
U=0\quad\Longleftrightarrow\quad
\operatorname{col}(W)\subseteq\operatorname{col}(W_q)
\quad\Longleftrightarrow\quad
\ker(W_q^\top)\subseteq\ker(W^\top).
$$

The first equivalence follows from $U=(I_n-P)W$; the second by taking orthogonal complements. Two inputs have the same quantized preactivation exactly when their difference $d$ satisfies $dW_q=0$. The kernel condition says every such $d$ must also satisfy $dW=0$: the FP output must be constant on each quantized fiber.

Conversely, if $U$ has a nonzero column $u$, take $d=u^\top$. Then $dW_q=0$, while $dW=dU$ has the positive entry $u^\top u$ in that column. So $x$ and $x+d$ have the same quantized preactivation but different FP preactivations, and no function of the quantized output alone can recover both when inputs are unrestricted. Figures 5, A1 and A2 show this with specific points; it does not require quantization to reduce rank. Recovering the output after ReLU, or recovering it only on a restricted set of inputs, is a different question.

### B.3 Bias and the sign convention in the code

With a float bias $b\in\mathbb R^{1\times m}$ shared by both layers, use the affine repair

$$
T(v)=vR+b-bR,\qquad z_q=xW_q+b,\qquad
T(z_q)+xU=xW+b.
$$

The path becomes $h\to T(h)\to\phi(T(z_q))\to y$, with

$$
c_R=T(h)-h,\qquad c_{\rm NL}=\phi(T(z_q))-T(h),\qquad
c_U=y-\phi(T(z_q)).
$$

These sum to $y-h$. Without an activation, $c_{\rm NL}$ is zero even with bias.

The code stores the opposite sign, **Q minus FP**:

| Stored field | Ledger term |
|---|---|
| `uncomp` | $-c_R$ |
| `nonlinear` | $-c_{\rm NL}$ |
| `range` (historical name) | $-c_U$ |
| `total` | $-c_{\rm total}$ |

The oracle adds FP minus Q, so it uses $c_{\rm total}$ directly. The stored network vectors `propagated` and local likewise have the opposite sign to $c_{\rm incoming}$ and $c_{\rm local}$ in Section 8.

With bias, the network comparison uses $h_q=\phi(aW_q+b)$, $g=\phi(aW+b)$ and $y_{\rm ref}=\phi(a_{\rm fp}W+b)$. The same local/incoming split follows by adding and subtracting $g$.

### B.4 The split depends on the choice of R

The ledger depends on the chosen $R$ and path. With the exact pseudoinverse, replacing $R$ by $R+K$ with $W_qK=0$ gives the same repaired map and the same $U$. But the affine repair becomes $T'(v)=T(v)+(v-b)K$. ReLU can move $h-b$ outside the row space of $W_q$, so $c_R$ and $c_{\rm NL}$ can change while their sum and the total stay the same. For example, with $W=W_q=\left[\begin{smallmatrix}1&-1\\0&0\end{smallmatrix}\right]$ and $x=(1,0)$, the identity-centered $R=I$ makes every term zero. Taking $K$ with every entry $0.5$ gives the same repaired map, but with $R'=I+K$, $hR'=(1.5,0.5)$: now $c_R=(0.5,0.5)$ and $c_{\rm NL}=(-0.5,-0.5)$, which cancel. We always use the identity-centered $R$ from B.1 when comparing terms.

### B.5 Interaction between terms: the Gram identity

For a dataset $\{x_k\}_{k=1}^N$, define

$$
G_{ij}=\frac1N\sum_k\langle c_i(x_k),c_j(x_k)\rangle,
\qquad i,j\in\{\mathrm{R},\mathrm{NL},\mathrm{U}\}.
$$

Terms pointing the same way add, opposite ones cancel, and perpendicular ones do not interact:

$$
\frac1N\sum_k\|c_{\rm total}(x_k)\|^2
=\sum_iG_{ii}+2\sum_{i<j}G_{ij}.
$$

For $C$ in Figure 7, the diagonal terms are $\|c_R\|^2=0.0296$, $\|c_{\rm NL}\|^2=0.0202$ and $\|c_U\|^2=0.0225$, summing to $0.0723$. The cross terms $\langle c_R,c_{\rm NL}\rangle=-0.0171$, $\langle c_R,c_U\rangle=-0.015$ and $\langle c_{\rm NL},c_U\rangle=-0.0038$ contribute $-0.0717$, leaving $\|c_{\rm total}\|^2=0.000625$. For $D$, the two cross terms involving $c_U$ change sign and the total is $0.275^2$.

The diagonal of $G$ holds the mean squared length of each term, and the off-diagonal entries their interactions. Flipping the sign of every term, as the code's convention does, leaves $G$ unchanged.

### B.6 Calibration set for Figure 9

Figure 9 uses $W_2$. Let $p$ be the 2D grid and circle points, $d=(1,-1,1)$ and $\alpha=(0.7,-0.7)^\top$. The inputs are

$$
x(p,t)=pW_q^\dagger+(p\alpha+t)d,\qquad
t\in\{-0.15,0,0.15\}.
$$

Since $dW_q=0$, the quantized preactivation is $p$ on all three planes. The figure shows the plane $t=0$; all three are used for fitting. The input matrix has rank $3$. Before ReLU, the offset along $d$ can be predicted from $p$, which gives $R_{\rm pre,data}=R+\alpha(dW)$ for this construction. ReLU changes $p$, so the fit after ReLU solves a different regression.

The exact matrices, objective values, input second moment, inputs and fit residuals are in [the comparison fixture](figures/quant_geometry/repair_vs_reconstruction_fixture.json). The paths and terms in the ledger figures are in [the ledger fixture](figures/quant_geometry/repair_ledger_figure_fixture.json).

<details>
<summary>Numerical note: a direction that is truly lost versus one dropped by truncation</summary>

Everything above uses the exact Moore–Penrose pseudoinverse. If the code drops small singular values, write the truncated pseudoinverse as $W_{q,\tau}^\dagger$ and use it in the formulas for $R$ and $U$. The resulting $R$ need not minimize the original objective. The decomposition $W=W_qR+U$ still holds, with

$$
P_\tau=W_qW_{q,\tau}^\dagger,\qquad
U_\tau=(I_n-P_\tau)(W-W_q).
$$

$U_\tau$ is only guaranteed to be orthogonal to the directions that were kept, and need not equal $(I_n-P_\tau)W$. A direction with a tiny **nonzero** singular value is still present in $xW_q$; recovering it may just need a large gain. So a nonzero $U_\tau$ shows what this choice of cutoff leaves uncorrected, not that information was lost. Figure A2 has a direction that is truly lost. A restricted set of inputs, or ReLU, can also hide the effect of a lost direction on the output.

For example, take $W=I$ and $W_q=\operatorname{diag}(1,10^{-8})$. A cutoff of $10^{-6}$ gives $R=I$ and $U=\operatorname{diag}(0,1-10^{-8})$. Keeping the small singular value instead gives $R=\operatorname{diag}(1,10^8)$ and $U=0$. The second input coordinate is weakly represented, not lost. In the truncated case $W_q^\top U=\operatorname{diag}(0,10^{-8}(1-10^{-8}))\ne0$.

See the [implementation README](quant_geometry_implementation.md#quantizer-and-decomposition) for cutoff settings.

</details>
