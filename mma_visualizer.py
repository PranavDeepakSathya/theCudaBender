import streamlit as st
import plotly.graph_objects as go

# -------------------------------
# Wiring law (source of truth)
# -------------------------------

def generate_inv_map_by_k(k, lc, rc):
    la = (k // 4) + (4 * (lc // 4))
    lb = (k // 4) + (8 * (lc % 4)) + (4 * (rc % 2))
    ra = (2 * ((k // 2) % 2)) + (rc // 2)
    rb = (k // 2) % 2
    pa = k % 2
    pb = k % 2
    return (la, ra, pa), (lb, rb, pb)


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(layout="wide")
st.title("MMA m16n8k16 – Structural Visualizer")

# Controls
L = 32
RC = 4

lc = st.slider("lc (lane)", 0, L - 1, 0)
rc = st.slider("rc (accumulator)", 0, RC - 1, 0)

# Generate contributors
A_pts = []
B_pts = []

for k in range(16):
    A, B = generate_inv_map_by_k(k, lc, rc)
    A_pts.append((*A, k))
    B_pts.append((*B, k))

# -------------------------------
# Plot A and B as true 3D lattices
# -------------------------------

fig = go.Figure()

# A operand
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in A_pts],
    y=[p[1] for p in A_pts],
    z=[p[2] for p in A_pts],
    mode='markers+text',
    marker=dict(size=6),
    text=[f"k={p[3]}" for p in A_pts],
    textposition="top center",
    name="A(la,ra,pa)",
))

# B operand
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in B_pts],
    y=[p[1] for p in B_pts],
    z=[p[2] for p in B_pts],
    mode='markers+text',
    marker=dict(size=6),
    text=[f"k={p[3]}" for p in B_pts],
    textposition="top center",
    name="B(lb,rb,pb)",
))

fig.update_layout(
    scene=dict(
        xaxis_title="lane",
        yaxis_title="reg",
        zaxis_title="pack",
    ),
    height=700,
    title=f"Contributors to C(lc={lc}, rc={rc})",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### How to read this

• Each point is a **register element** (not flattened, not reshaped)

• Axes are literal:
- x = lane index
- y = register index
- z = pack index

• Each highlighted point corresponds to one **k ∈ [0..15]**

• Clicking different (lc, rc) shows how wiring shifts structurally

This visualization is driven **only** by the affine wiring law.
No tables. No inverse maps. No heuristics.
""")
