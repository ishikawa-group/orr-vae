#!/usr/bin/env python3
import os, torch, matplotlib.pyplot as plt, ase.io
from pathlib import Path
from ase.build import fcc100
from catalyst_ccVAE import CVAE
from tool import tensor_to_slab, sort_atoms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------- 連番(1-3) → 元の原子番号(47/78/79) への逆 LUT -------------
REV_LUT = torch.tensor([0, 47, 78, 79], dtype=torch.int64)   # index 0 未使用

def load_model(path:Path, latent:int=64) -> CVAE:
    m = CVAE(latent, condition_dim=1).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

def sorted_template(size=(4,4,4), a=4.0):
    slab = fcc100("Pt", size=size, a=a, vacuum=None, periodic=True)
    return sort_atoms(slab, axes=("z","y","x"))               # ソート済み

@torch.no_grad()
def generate_slab(model:CVAE, overpot:float, z=None, tmpl=None):
    if tmpl is None: tmpl = sorted_template()
    if z is None:   z = torch.randn(1, model.latent_size, device=device)

    y = torch.tensor([[overpot]], dtype=torch.float32, device=device)
    z_cat = torch.cat([z, model.label_encoder(y)], 1)
    tensor_f = model.decoder(z_cat)[0]          # (4,8,8) float32 (1-3)

    # ------ 1-3 → 47/78/79 に戻す -----------------------------
    idx = torch.round(tensor_f).clamp(1,3).to(torch.int64)   # 1–3
    tensor_Z = REV_LUT[idx]                                  # 原子番号テンソル

    slab = tensor_to_slab(tensor_Z, tmpl)                    # ASE Atoms
    return slab, tensor_Z

def save_tensor_img(tensor, path, title="tensor"):
    fig, ax = plt.subplots(1,4,figsize=(14,3))
    for i in range(4):
        im = ax[i].imshow(tensor[i], cmap="viridis")
        ax[i].set_title(f"layer {i}")
        ax[i].axis("off")
    plt.colorbar(im, ax=ax.ravel().tolist())
    plt.suptitle(title); plt.tight_layout(); plt.savefig(path); plt.close()

def main():
    out_dir = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/generated_catalysts")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/ccVAE/final_model.pt"))
    tmpl   = sorted_template()

    for i, op in enumerate([0.3,0.4,0.5,0.6,0.7]):
        torch.manual_seed(42+i)               # 再現性
        slab, tensor_Z = generate_slab(model, op, tmpl=tmpl)

        ase.io.write(out_dir/f"catalyst_op_{op:.2f}.cif", slab)
        ase.io.write(out_dir/f"catalyst_op_{op:.2f}.png", slab)
        save_tensor_img(tensor_Z.numpy(), out_dir/f"tensor_op_{op:.2f}.png",
                        f"OP {op:.2f} V")

        print(f"✓ {op:.2f} V 生成完了 → CIF & PNG 保存")

if __name__ == "__main__":
    main()
