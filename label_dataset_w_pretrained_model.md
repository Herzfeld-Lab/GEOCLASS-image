# Labeling a Dataset with a Pretrained Model — Notes

## How `test.py` actually wires up `VarioNet`

When `cfg['model'] == 'VarioNet'`, three weight files must all line up architecturally:

1. `vario_mlp.pth` (or `12vario_mlp.pth`) — pretrained `VarioMLP` submodel.
2. `resnet18.pth` (or `12resnet18.pth`) — pretrained `Resnet18` submodel.
3. The `--load_checkpoint epoch_*` file — the fine-tuned joint `VarioNet` checkpoint.

Steps 1 and 2 build empty modules sized from the config (`num_classes`, `vario_num_lag`, `hidden_layers`) and load those files into them. Step 3 then **overwrites** both submodels' weights via `model.load_state_dict(checkpoint['state_dict'])`. So the `.pth` files mostly act as architectural scaffolding — their values get replaced, but their *shapes* must match.

The relative paths in `torch.load('vario_mlp.pth', ...)` are resolved against the current working directory, so the files must live at the **repo root** (`GEOCLASS-image/`). Other scripts that touch VarioNet (`train.py`, `train_only_varnet.py`, `test_varnet.py`, `train_res_and_var.py`) use the same convention.

`train_res_and_var.py` is the script that produces these `.pth` files (lines 300–301 and 369–370).

## How `VarioMLP` shape derives from the config

```17:31:Models/VarioMLP.py
        self.input_size = vario_num_lag * 3
        self.output_size = num_classes
        self.hidden_size = [int(i * self.input_size) for i in hidden_layers]
        self.num_lag = vario_num_lag

        self.input = nn.Linear(self.input_size, self.hidden_size[0])
        self.lrelu = nn.LeakyReLU()
        #self.lrelu = nn.Tanh()

        self.hidden = nn.ModuleList()

        for i in range(len(hidden_layers) - 1):
            self.hidden.append(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))

        self.output = nn.Linear(self.hidden_size[-1], self.output_size)
```

For `vario_num_lag = N`, `hidden_layers = [a, b]`, `num_classes = C`:

- `input.weight = [a*3N, 3N]`
- `hidden.0.weight = [b*3N, a*3N]`
- `output.weight = [C, b*3N]`

Working backwards from a state_dict's `input.weight` shape always recovers `vario_num_lag`. Useful sanity-check from the repo root:

```bash
python -c "import torch; sd=torch.load('12vario_mlp.pth', map_location='cpu'); print(sd['input.weight'].shape)"
```

For `vario_num_lag: 33, hidden_layers: [5,2], num_classes: 13` it should print `torch.Size([495, 99])`. For `vario_num_lag: 53` it would be `[795, 159]`.

## Three errors encountered and their causes

### 1. `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`

**Cause:** `torch.load(...)` was called without `map_location`, and the checkpoint's tensors were pickled with `cuda:0` device tags. On a CPU-only Mac, deserialization tries to put them back on `cuda:0` and fails.

**Important:** `use_cuda: False` in the config does **not** prevent this. `test.py` reads CUDA only from the `-c/--cuda` command-line flag, not the config; and even `args.cuda` doesn't affect how `torch.load` deserializes a GPU-saved checkpoint.

**Fix:** add `map_location=torch.device('cpu')` to all three `torch.load(...)` calls. Implemented as a `load_map_location` variable gated on the config's `use_cuda` field, so the behavior is opt-in:

```37:41:test.py
imgTrain = cfg['train_with_img']
adapt = cfg['adaptive']

# When use_cuda is False, force checkpoints saved on a GPU to load onto CPU.
load_map_location = None if cfg.get('use_cuda', False) else torch.device('cpu')
```

### 2. State-dict size mismatch on the joint `epoch_49` checkpoint

**Cause:** the config's `vario_num_lag` did not match the architecture the joint checkpoint was trained with. The error message itself reveals both: "copying a param with shape `[495, 99]` from checkpoint, the shape in current model is `[795, 159]`" → the saved checkpoint was 33-lag, the current config was 53-lag.

**Fix:** set `vario_num_lag: 33` in `Config/greenland_enthalpy/jakobshavn-wv/mlp_negri12_for_jak.config` to match the original training config used to produce that checkpoint (`Config/mlp_12_negri/mlp_test_negri12.config`).

### 3. State-dict size mismatch on `12vario_mlp.pth`

**Cause:** after the config fix, `vario_mlp.pth` / `12vario_mlp.pth` at the repo root must also be 33-lag. The file dropped in was a 53-lag version (input.weight `[795, 159]`).

**Fix:** find the matching 33-lag `.pth` (sanity-check with the one-liner above) and place it in the repo root with the filename the script expects. If no original 33-lag `.pth` is recoverable, re-running `train_res_and_var.py` with `vario_num_lag: 33` regenerates compatible scaffolding — the joint checkpoint will overwrite the values anyway.

## Does the `_split.npy` need to be remade when changing `vario_num_lag`?

**No.** The `.npy` only stores split-image metadata (locations, `winsize_pix`) and labels. Variograms are computed at runtime by the `DirectionalVario(vario_num_lag)` transform attached to the dataset:

```527:539:Dataset.py
class DirectionalVario(object):

    def __init__(self, numLag):
        self.numLag = numLag

    def __call__(self, img):
        imSize = img.shape
        if (imSize[0] == 201 and imSize[1] == 268) or (imSize[0] == 268 and imSize[1] == 201):
            return silas_directional_vario(img, self.numLag)

        else:
            print("Use an image size of (201,268) for best results")
            return fast_directional_vario(img, self.numLag)
```

The filename convention `..._<N>_(201,268)_split.npy` encodes the count of split images and `split_img_size`, but not the lag count, which is consistent with the lag count being a runtime parameter only.

The `.npy` only needs to be rebuilt if `split_img_size`, `img_path`, or `contour_path` change.

## Cross-machine notes (Intel Mac on Ventura 13.0.1)

- PyTorch dropped `osx-x86_64` wheels starting with v2.3 (early 2024). On Intel Mac the last installable version is **`torch==2.2.2` / `torchvision==0.17.2`**, and it requires Python 3.8–3.11.
- `pip install torch` failing with "from versions: none" is platform-tag mismatch, usually compounded by Python being too new (e.g. 3.14 has no torch wheel for Intel macOS at all).
- Conda envs are **not** portable between architectures. The current Mac is likely `osx-arm64`; the destination is `osx-64`. Use `conda env export --from-history > environment.yml`, then on the Intel Mac create a fresh env with Python 3.11 and pin `torch==2.2.2`.
- The Radeon Pro 5500 XT has no PyTorch backend — Intel Mac will be CPU-only. The `map_location='cpu'` fix in `test.py` is required there too.

## Quick checklist for "label a dataset with a pretrained VarioNet checkpoint"

1. Locate the **training config** that produced the joint `epoch_*` checkpoint (usually saved alongside it on disk). Note its `num_classes`, `vario_num_lag`, `hidden_layers`, `split_img_size`.
2. Make a test config that matches all four of those fields, with the dataset paths pointed at your new region.
3. Place 33-lag (or whatever-lag) `vario_mlp.pth` and `resnet18.pth` (or the `12*` variants the script references) at the **repo root**. Verify their `input.weight` shape with the one-liner.
4. If running on CPU only, leave `use_cuda: False` in the config so `load_map_location` kicks in. Do **not** pass `-c` on the CLI.
5. Run:

```bash
python3 test.py <your.config> --load_checkpoint <path/to/checkpoints/epoch_N>
```

6. Output is written by `np.save(output_dir+"/labels/labeled_"+checkpoint_str, dataset)` (test.py line 264) — `output_dir` is derived as `split[0]+'/'+split[1]` of the checkpoint path, so the labels land in `<split[0]>/<split[1]>/labels/`.