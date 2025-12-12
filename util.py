import math
import torch
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


def cal_loss(pred, ground_truth, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    ground_truth = ground_truth.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, ground_truth.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, ground_truth, reduction='mean')

    return loss


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query.

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    
    Output:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(k, xyz, new_xyz):
    """
    K nearest neighborhood.

    Input:
        k: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    
    Output:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    
    Output:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_ball_group(s, radius, n, coords, features):
    """
    Sampling by FPS and grouping by ball query.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by ball query
        n[int]: fix number of points in ball neighbor
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    new_features = index_points(features, fps_idx)                     # [B, s, D]

    # ball_query grouping
    idx = query_ball_point(radius, n, coords, new_coords)              # [B, s, n]
    grouped_features = index_points(features, idx)                     # [B, s, n, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, n, D]

    # Concat, my be different in many networks
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, n, 1)], dim=-1)  # [B, s, n, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, n, 2D]


def sample_and_knn_group(s, k, coords, features):
    """
    Sampling by FPS and grouping by KNN.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    new_features = index_points(features, fps_idx)                     # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, coords, new_coords)                                              # [B, s, k]
    grouped_features = index_points(features, idx)                                      # [B, s, k, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)  # [B, s, k, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, k, 2D]


def _kdtree_partition(points_xyz, leaf_size):
    """Partition points on the same device using KD-style binary splits."""
    device = points_xyz.device
    N = points_xyz.shape[0]
    if N == 0:
        return []
    stack = [torch.arange(N, device=device, dtype=torch.long)]
    leaves = []
    while stack:
        idx = stack.pop()
        if idx.numel() <= leaf_size:
            leaves.append(idx)
            continue
        pts = points_xyz[idx]
        ranges = pts.max(dim=0).values - pts.min(dim=0).values
        split_dim = int(torch.argmax(ranges).item())
        values = pts[:, split_dim]
        _, order = torch.sort(values)
        mid = max(1, order.numel() // 2)
        left = idx[order[:mid]]
        right = idx[order[mid:]]
        stack.append(right)
        stack.append(left)
    return leaves


def _allocate_samples_per_leaf(leaves, total_samples):
    sizes = [len(lf) for lf in leaves]
    total_points = sum(sizes)
    if total_points == 0:
        return [0] * len(leaves)
    raw = [size * total_samples / total_points for size in sizes]
    base = [max(1, int(math.floor(x))) for x in raw]
    current = sum(base)
    # adjust down if we overshoot
    while current > total_samples:
        idx = max((i for i in range(len(base)) if base[i] > 1), key=lambda i: base[i], default=None)
        if idx is None:
            break
        base[idx] -= 1
        current -= 1
    # adjust up if we undershoot
    remainders = [r - math.floor(r) for r in raw]
    while current < total_samples and base:
        idx = max(range(len(base)), key=lambda i: remainders[i])
        base[idx] += 1
        remainders[idx] = 0.0
        current += 1
    return base


def _fps_indices(points, m):
    """Leaf-level FPS using pointnet2_ops (points: [L,3]). Returns indices [m]."""
    L = points.shape[0]
    device = points.device
    if L == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if m >= L:
        return torch.arange(L, device=device, dtype=torch.long)
    pts = points.unsqueeze(0).contiguous()  # [1, L, 3]
    idx = pointnet2_utils.furthest_point_sample(pts, m).squeeze(0).long()  # [m]
    return idx


def _sample_leaf_indices(leaf, count, coords, strategy):
    if leaf.numel() == 0 or count <= 0:
        return leaf.new_empty(0)
    if strategy == 'fps':
        pts = coords[leaf, :]
        base = _fps_indices(pts, min(count, leaf.numel()))
        chosen = leaf[base]
    else:
        if count >= leaf.numel():
            chosen = leaf
        else:
            perm = torch.randperm(leaf.numel(), device=leaf.device)[:count]
            chosen = leaf[perm]
    if chosen.numel() < count:
        extra = chosen[torch.randint(0, max(1, chosen.numel()), (count - chosen.numel(),), device=leaf.device)]
        chosen = torch.cat([chosen, extra], dim=0)
    elif chosen.numel() > count:
        perm = torch.randperm(chosen.numel(), device=leaf.device)[:count]
        chosen = chosen[perm]
    return chosen


def _ensure_sample_count(indices, leaf_ids, target, total_points):
    if indices.numel() == 0:
        if total_points == 0:
            return indices, leaf_ids
        base = torch.arange(min(target, total_points), device=leaf_ids.device if leaf_ids.numel() else indices.device, dtype=torch.long)
        if base.numel() < target:
            extra = base[torch.randint(0, base.numel(), (target - base.numel(),), device=base.device)]
            base = torch.cat([base, extra], dim=0)
        leaf_ids = torch.zeros(base.numel(), dtype=torch.long, device=base.device)
        return base[:target], leaf_ids[:target]
    if indices.numel() >= target:
        perm = torch.randperm(indices.numel(), device=indices.device)[:target]
        return indices[perm], leaf_ids[perm]
    deficit = target - indices.numel()
    extra_sel = torch.randint(0, indices.numel(), (deficit,), device=indices.device)
    indices = torch.cat([indices, indices[extra_sel]], dim=0)
    leaf_ids = torch.cat([leaf_ids, leaf_ids[extra_sel]], dim=0)
    return indices, leaf_ids


def _sample_neighbors_for_leaf(leaf, k):
    if leaf.numel() == 0:
        return leaf.new_zeros(k)
    if leaf.numel() >= k:
        perm = torch.randperm(leaf.numel(), device=leaf.device)[:k]
        return leaf[perm]
    reps = leaf[torch.randint(0, leaf.numel(), (k - leaf.numel(),), device=leaf.device)]
    return torch.cat([leaf, reps], dim=0)


def kdtree_sample_centers(s, coords, leaf_size=64, strategy='random'):
    """Return KD-selected center indices per batch: coords [B,N,3] -> indices [B,s]."""
    device = coords.device
    B, N, _ = coords.shape
    center_indices_all = []
    for b in range(B):
        pts = coords[b]
        leaves = _kdtree_partition(pts, leaf_size)
        if not leaves:
            leaves = [torch.arange(N, device=device, dtype=torch.long)]
        leaf_samples = _allocate_samples_per_leaf([leaf.tolist() for leaf in leaves], s)
        centers_list = []
        for leaf, quota in zip(leaves, leaf_samples):
            if quota <= 0 or leaf.numel() == 0:
                continue
            selected = _sample_leaf_indices(leaf, quota, pts, 'fps' if strategy != 'random' else 'random')
            centers_list.append(selected)
        if centers_list:
            centers = torch.cat(centers_list)
        else:
            centers = torch.empty(0, dtype=torch.long, device=device)
        leaf_ids_dummy = torch.zeros(centers.numel(), dtype=torch.long, device=device)
        centers, _ = _ensure_sample_count(centers, leaf_ids_dummy, s, N)
        center_indices_all.append(centers.unsqueeze(0))
    return torch.cat(center_indices_all, dim=0)


def sample_and_kdtree_simple_group(s, k, coords, features, leaf_size=64, strategy='random'):
    """KD-tree partition + per-leaf sampling (random/FPS) to reach sampling rate."""
    device = coords.device
    B, _, _ = coords.shape
    grouped_dim = features.shape[-1] * 2
    new_coords_list = []
    new_features_list = []
    for b in range(B):
        pts = coords[b]
        feats = features[b]
        leaves = _kdtree_partition(pts, leaf_size)
        if not leaves:
            leaves = [torch.arange(pts.shape[0], device=device, dtype=torch.long)]
        leaf_samples = _allocate_samples_per_leaf([leaf.tolist() for leaf in leaves], s)
        center_indices = []
        center_leaf_ids = []
        for lid, (leaf, quota) in enumerate(zip(leaves, leaf_samples)):
            if quota <= 0 or leaf.numel() == 0:
                continue
            selected = _sample_leaf_indices(leaf, quota, pts, 'fps' if strategy != 'random' else 'random')
            center_indices.append(selected)
            center_leaf_ids.append(torch.full((selected.numel(),), lid, device=device, dtype=torch.long))
        if center_indices:
            centers = torch.cat(center_indices)
            leaf_ids = torch.cat(center_leaf_ids)
        else:
            centers = torch.empty(0, dtype=torch.long, device=device)
            leaf_ids = torch.empty(0, dtype=torch.long, device=device)
        centers, leaf_ids = _ensure_sample_count(centers, leaf_ids, s, pts.shape[0])
        center_coords = pts[centers]
        center_feats = feats[centers]
        grouped = torch.empty(centers.numel(), k, grouped_dim, device=device, dtype=feats.dtype)
        order_idx = torch.arange(centers.numel(), device=device, dtype=torch.long)
        unique_leaf_ids = torch.unique(leaf_ids, sorted=False)
        for lid in unique_leaf_ids:
            mask = leaf_ids == lid
            pos = order_idx[mask]
            rep_feats = center_feats[pos]
            neighbor_idx = _sample_neighbors_for_leaf(leaves[lid], k)
            neigh_feats = feats[neighbor_idx].unsqueeze(0).expand(rep_feats.size(0), -1, -1)
            center_expand = rep_feats.unsqueeze(1).expand(-1, k, -1)
            grouped[pos] = torch.cat([neigh_feats - center_expand, center_expand], dim=-1)
        new_coords_list.append(center_coords.unsqueeze(0))
        new_features_list.append(grouped.unsqueeze(0))
    new_coords = torch.cat(new_coords_list, dim=0)
    new_features = torch.cat(new_features_list, dim=0)
    return new_coords, new_features


class Logger():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


if __name__ == '__main__':
    points = torch.rand(32, 1024, 3).to('cuda')
    features = torch.rand(32, 1024, 128).to('cuda')
    new_points, new_features = sample_and_knn_group(512, 32, points, features)
    print(new_points.size())
    print(new_features.size())
