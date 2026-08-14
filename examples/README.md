# Examples

The five public examples are stored in `demo/images/`. Run the command in the
repository root:

```bash
python inference/run_demo.py \
  --weights weights/Banana-Gpose-best.pt \
  --source demo/images \
  --output demo/results
```

Each input produces a binary fruit-stalk mask, fruit-region-prior overlay,
Growth-auto optimization overlay, signed posture result, and final visualization.
