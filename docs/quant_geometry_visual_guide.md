# Quantization geometry reading guide

The visual explanation and formal definitions live together in [Quantization geometry: deformation, repair, and an exact error ledger](quant_geometry_foundations.md). Start there: each ledger component has its definition, an explanation, and a picture of the exact states it compares.

- [Maps and geometric deformation](quant_geometry_foundations.md#1-what-shape-does-the-layer-move)
- [Map repair, calibration, and the ReLU obstruction](quant_geometry_foundations.md#2-which-part-can-a-map-on-the-quantized-output-recover)
- [Three correction objectives and their geometric comparison](quant_geometry_foundations.md#fitting-the-clipped-outputs-answers-a-different-question)
- [The four-state ledger and its three components](quant_geometry_foundations.md#3-the-ledger-four-states-three-signed-displacements)
- [Signed cancellation and interaction](quant_geometry_foundations.md#4-why-component-magnitudes-cannot-be-added)
- [Creation, propagation, and oracle interventions](quant_geometry_foundations.md#5-creation-here-versus-propagation-through-the-network)
- [Supplementary examples of the remainder U](quant_geometry_foundations.md#a-understanding-the-remainder-u)
- [Technical derivations and conventions](quant_geometry_foundations.md#b-technical-details)

For the moving version, see the animation reproduction instructions in the [implementation README](quant_geometry_implementation.md#reproduce-the-artifacts). Tests, results, diagnostic figures, mathematical audit, and contribution records are in the [verification report](quant_geometry_verification.md).
