"""
bookshelf_writer.py — Convert a protobuf benchmark to ICCAD-style
Bookshelf format files that DREAMPlace can consume.

Bookshelf format (the one ICCAD-style placers expect):
  *.aux:    one-line file listing the other files
  *.nodes:  node list (name, width, height, [terminal])
  *.nets:   net connectivity (one block per net)
  *.pl:     initial placement (x, y, orient, [/FIXED])
  *.wts:    net weights (optional)
  *.scl:    site/row info (one big row covering the canvas)

For ICCAD04, all macros are heterogeneous in size. We model hard macros as
fixed-orientation movable nodes; soft macros are treated similarly (just
larger blocks). The canvas becomes one giant row covering [0,canvas_h].

IMPORTANT: DREAMPlace's Limbo Bookshelf parser requires INTEGER values
for node widths/heights, positions, pin offsets, and site counts. ICCAD04
microns are fractional (e.g., a 0.84-micron macro), so we scale every
coordinate by `SCALE` (1000) on write. The placer must apply the inverse
scale when reading DREAMPlace's output `.pl` back into our framework.
"""

import math
import os
from pathlib import Path
import numpy as np

# Scale factor: multiply by this when writing bookshelf, divide by this when
# reading the placement back. 1000× gives us 3 fractional digits of
# precision and keeps all values comfortably inside int32 range for ICCAD04
# (largest canvas dim ~80 → ~80000 after scaling).
SCALE = 1000


def write_bookshelf(out_dir, name, benchmark, plc, init_positions):
    """
    Write Bookshelf files describing the benchmark into `out_dir`.
    `init_positions`: [n_hard, 2] np.ndarray with starting (x, y) per macro.

    Returns the path to the .aux file (entry point for DREAMPlace).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_hard = benchmark.num_hard_macros
    n_macros_total = benchmark.num_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.numpy()
    pos = init_positions
    movable = benchmark.get_movable_mask().numpy()

    # Local helper: round a fractional micron to scaled integer DBU.
    def _q(v):
        return int(round(float(v) * SCALE))

    # I/O ports — referenced by net_nodes with indices in
    # [n_macros_total, n_macros_total + n_ports). They must appear in .nodes
    # (as fixed terminals) and .pl, or DREAMPlace's Bookshelf parser rejects
    # the .nets file when it sees references like o5000.
    port_positions = benchmark.port_positions.numpy() if benchmark.port_positions.numel() else np.zeros((0, 2))
    n_ports = int(port_positions.shape[0])
    total_nodes = n_macros_total + n_ports

    aux_path = out_dir / f"{name}.aux"
    nodes_path = out_dir / f"{name}.nodes"
    nets_path = out_dir / f"{name}.nets"
    pl_path = out_dir / f"{name}.pl"
    wts_path = out_dir / f"{name}.wts"
    scl_path = out_dir / f"{name}.scl"

    # ── .aux ─────────────────────────────────────────────────────────────────
    with open(aux_path, "w") as f:
        f.write(f"RowBasedPlacement : {name}.nodes {name}.nets {name}.wts {name}.pl {name}.scl\n")

    # ── .nodes ───────────────────────────────────────────────────────────────
    # NumNodes = movable macros + fixed macros + ports.
    # NumTerminals = fixed macros + ports (all ports are treated as fixed pins).
    n_fixed_macros = int(np.sum(~movable[:n_macros_total]))
    n_terms = n_fixed_macros + n_ports
    with open(nodes_path, "w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {total_nodes}\n")
        f.write(f"NumTerminals : {n_terms}\n\n")
        # Bookshelf convention: movable nodes first, then terminals.
        # We preserve our integer indices so net references stay valid:
        # write nodes in original index order regardless. Most parsers
        # (including DREAMPlace's Limbo) accept this.
        for i in range(n_macros_total):
            w = max(1, _q(sizes[i, 0]))
            h = max(1, _q(sizes[i, 1]))
            terminal = "" if movable[i] else " terminal"
            f.write(f"  o{i}  {w}  {h}{terminal}\n")
        # Ports: tiny fixed nodes at their canvas positions (1×1 DBU each).
        for p in range(n_ports):
            f.write(f"  o{n_macros_total + p}  1  1 terminal\n")

    # ── .nets ────────────────────────────────────────────────────────────────
    # Each net lists its connected nodes. Pin offsets are relative to the
    # node's lower-left corner in Bookshelf; emit per-pin offsets when
    # net_pin_nodes is populated, fall back to 0,0 (node center for ports,
    # macro lower-left for macros — close enough to optimize WL on ICCAD04).
    net_nodes = benchmark.net_nodes
    net_pin_nodes = benchmark.net_pin_nodes
    macro_pin_offsets = benchmark.macro_pin_offsets
    n_nets = len(net_nodes)

    # Compute total pins: if pin-level data exists use that (preserves
    # multiple pins on same macro); otherwise per-node uniques from net_nodes.
    have_pin_level = bool(net_pin_nodes) and len(net_pin_nodes) == n_nets
    if have_pin_level:
        total_pins = sum(int(npn.shape[0]) for npn in net_pin_nodes)
    else:
        total_pins = sum(len(nn) for nn in net_nodes)

    # Bookshelf pin offsets are relative to the node's LOWER-LEFT corner
    # and must be INTEGER DBU. Our macro_pin_offsets come from
    # PlacementCost / Google CT and are CENTER-relative in microns.
    # Convert: dx_bookshelf = round((dx_center + w/2) * SCALE).
    def _node_size_microns(idx):
        if idx < n_macros_total:
            return float(sizes[idx, 0]), float(sizes[idx, 1])
        return 1.0 / SCALE, 1.0 / SCALE  # port = 1 DBU = 1/SCALE microns

    with open(nets_path, "w") as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {n_nets}\n")
        f.write(f"NumPins : {total_pins}\n\n")
        for k in range(n_nets):
            if have_pin_level:
                pin_rows = net_pin_nodes[k].numpy()  # [num_pins, 2] (owner, slot)
                deg = int(pin_rows.shape[0])
                f.write(f"NetDegree : {deg}  n{k}\n")
                for r in range(deg):
                    owner = int(pin_rows[r, 0])
                    slot = int(pin_rows[r, 1])
                    w_i, h_i = _node_size_microns(owner)
                    if owner < n_hard:
                        offs = macro_pin_offsets[owner] if owner < len(macro_pin_offsets) else None
                        if offs is not None and slot < offs.shape[0]:
                            dx_c = float(offs[slot, 0])
                            dy_c = float(offs[slot, 1])
                        else:
                            dx_c, dy_c = 0.0, 0.0
                    else:
                        # Soft macros & ports: pin at node center (no offsets).
                        dx_c, dy_c = 0.0, 0.0
                    # Convert center-relative → lower-left-relative, then scale.
                    dx = _q(dx_c + w_i / 2)
                    dy = _q(dy_c + h_i / 2)
                    f.write(f"  o{owner}  I  : {dx}  {dy}\n")
            else:
                nodes = net_nodes[k]
                deg = len(nodes)
                f.write(f"NetDegree : {deg}  n{k}\n")
                for ni in nodes:
                    idx = int(ni)
                    w_i, h_i = _node_size_microns(idx)
                    f.write(f"  o{idx}  I  : {_q(w_i/2)}  {_q(h_i/2)}\n")

    # ── .pl (initial placement) ──────────────────────────────────────────────
    # Bookshelf uses lower-left corner coords; our positions are centers.
    # All positions written as integer DBU (× SCALE).
    with open(pl_path, "w") as f:
        f.write("UCLA pl 1.0\n\n")
        all_positions = benchmark.macro_positions.numpy()
        out_pos = all_positions.copy()
        out_pos[:n_hard] = pos
        for i in range(n_macros_total):
            w = float(sizes[i, 0])
            h = float(sizes[i, 1])
            x = float(out_pos[i, 0]) - w / 2
            y = float(out_pos[i, 1]) - h / 2
            fixed = " /FIXED" if not movable[i] else ""
            f.write(f"  o{i}  {_q(x)}  {_q(y)}  : N{fixed}\n")
        # Ports: 1-DBU fixed nodes positioned at their port_positions
        # (port_position is the pin center; node lower-left = center - 0.5 DBU).
        for p in range(n_ports):
            px = float(port_positions[p, 0])
            py = float(port_positions[p, 1])
            f.write(f"  o{n_macros_total + p}  {_q(px)}  {_q(py)}  : N /FIXED\n")

    # ── .wts ─────────────────────────────────────────────────────────────────
    with open(wts_path, "w") as f:
        f.write("UCLA wts 1.0\n\n")
        for k in range(n_nets):
            f.write(f"n{k} 1\n")

    # ── .scl (sites/rows) ────────────────────────────────────────────────────
    # One single row covering the whole canvas (now in scaled DBU).
    # DREAMPlace's nonlinear placer doesn't constrain macros to rows.
    #
    # ICCAD04 convention: Coordinate/Height/Sitewidth/Sitespacing are
    # integers; Siteorient/Sitesymmetry are letters (N/Y).
    n_sites = max(1, int(math.ceil(cw * SCALE)))
    row_height = max(1, int(math.ceil(ch * SCALE)))
    with open(scl_path, "w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write("NumRows : 1\n\n")
        f.write("CoreRow Horizontal\n")
        f.write(f"  Coordinate    :   0\n")
        f.write(f"  Height        :   {row_height}\n")
        f.write(f"  Sitewidth     :   1\n")
        f.write(f"  Sitespacing   :   1\n")
        f.write(f"  Siteorient    :   N\n")
        f.write(f"  Sitesymmetry  :   Y\n")
        f.write(f"  SubrowOrigin  :   0\tNumSites :   {n_sites}\n")
        f.write("End\n")

    return aux_path


def read_dreamplace_pl(pl_path, n_hard):
    """
    Parse a DREAMPlace output .pl file. Returns [n_hard, 2] np.ndarray of
    (x_center, y_center) for hard macros (assumed indices 0..n_hard-1).
    """
    centers = np.zeros((n_hard, 2), dtype=np.float64)
    seen = np.zeros(n_hard, dtype=bool)
    with open(pl_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("UCLA") or line.startswith("#") or ":" not in line:
                continue
            # Format: "  o0   123.45   67.89   : N /FIXED"
            parts = line.split()
            if len(parts) < 4 or not parts[0].startswith("o"):
                continue
            try:
                idx = int(parts[0][1:])
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue
            if idx < n_hard:
                # x, y in .pl are lower-left; we need center. But the writer
                # also stored lower-left. We need the macro size to convert.
                # That's added in placer.py where sizes are available.
                centers[idx, 0] = x
                centers[idx, 1] = y
                seen[idx] = True
    return centers, seen
