import numpy as np
import plotly.graph_objects as go
import pickle

def pointcloud(
    T_chart_points: np.ndarray, downsample=5, colors=None, scene="scene", name=None
) -> go.Scatter3d:
    marker_dict = {"size": 3}
    if colors is not None:
        try:
            a = [f"rgb({r}, {g}, {b})" for r, g, b in colors][::downsample]
            marker_dict["color"] = a
        except:
            marker_dict["color"] = colors[::downsample]
    return go.Scatter3d(
        x=-T_chart_points[0, ::downsample],
        y=T_chart_points[2, ::downsample],
        z=T_chart_points[1, ::downsample],
        mode="markers",
        marker=marker_dict,
        scene=scene,
        name=name,
        showlegend=False,
    )

def _flow_traces_v2(
    pos, flows, sizeref=0.05, scene="scene", flowcolor="red", name="flow"
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]

    n_dest = n_pos + n_flows * sizeref

    for i in range(len(n_pos)):
        x_lines.append(-n_pos[i][0])
        y_lines.append(n_pos[i][2])
        z_lines.append(n_pos[i][1])
        x_lines.append(-n_dest[i][0])
        y_lines.append(n_dest[i][2])
        z_lines.append(n_dest[i][1])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=10),
        name=name,
        showlegend=False,
    )

    # norm_flows = n_flows / n_flows.norm(dim=-1).unsqueeze(-1)

    # cones_trace = go.Cone(
    #     x=n_dest[:, 0],
    #     y=n_dest[:, 1],
    #     z=n_dest[:, 2],
    #     u=norm_flows[:, 0],
    #     v=norm_flows[:, 1],
    #     w=norm_flows[:, 2],
    #     colorscale="Blues",
    #     sizemode="scaled",
    #     showscale=False,
    #     sizeref=1.0,
    #     scene=scene,
    # )
    head_trace = go.Scatter3d(
        x=-n_dest[:, 0],
        y=n_dest[:, 2],
        z=n_dest[:, 1],
        mode="markers",
        marker={"size": 3, "color": "darkred"},
        scene=scene,
        showlegend=False,
    )

    return [lines_trace, head_trace]

def _3d_scene(data):
    # Create a 3D scene which is a cube w/ equal aspect ratio and fits all the data.

    assert data.shape[1] == 3
    # Find the ranges for visualizing.
    mean = data.mean(axis=0)
    max_x = np.abs(data[:, 0] - mean[0]).max()
    max_y = np.abs(data[:, 1] - mean[1]).max()
    max_z = np.abs(data[:, 2] - mean[2]).max()
    all_max = max(max(max_x, max_y), max_z)
    scene = dict(
        xaxis=dict(nticks=10, range=[mean[0] - all_max, mean[0] + all_max]),
        yaxis=dict(nticks=10, range=[mean[1] - all_max, mean[1] + all_max]),
        zaxis=dict(nticks=10, range=[mean[2] - all_max, mean[2] + all_max]),
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene

def _3d_scene_fixed(x_range, y_range, z_range):
    scene = dict(
        xaxis=dict(nticks=10, range=x_range),
        yaxis=dict(nticks=10, range=y_range),
        zaxis=dict(nticks=10, range=z_range),
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene

def create_flow_frame(pts, flow):
    f = go.Figure()
    f.add_trace(pointcloud(pts.T, downsample=1, scene="scene1"))
    ts = _flow_traces_v2(pts, flow, sizeref=0.01, scene="scene1")
    for t in ts:
        f.add_trace(t)
    f.update_layout(scene1=_3d_scene(pts))
    # f.update_layout(scene1=_3d_scene_fixed([-0.25, 0.25], [-0.25, 0.25], [0.0, 0.65]))

    # f.write_html(osp.join(savedir, f'{i}_tmp.html'))
    f.write_html("flow_frame.html")
    # fig_bytes = f.to_image(format="png", engine="kaleido")
    # buf = io.BytesIO(fig_bytes)
    # img = Image.open(buf)
    # return np.asarray(img)

if __name__ == '__main__':
    with open('dummy_data/dummy_0.2d_0.025r.pkl', 'rb') as f:
        data = pickle.load(f)

    create_flow_frame(data['points'][1], data['flow'][1])
