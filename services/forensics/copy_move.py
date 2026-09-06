from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import cv2
import numpy as np


class CopyMoveDetector:
    """Detect duplicated regions (copy-move forgery) via ORB keypoint
    self-matching with shift-vector clustering, and a SIFT fallback for
    cases ORB misses.

    v3 (ADR-026). The v2 DCT-grid detector sampled blocks on a fixed 16 px
    stride, so a duplication whose offset was not a multiple of the stride
    never produced comparable blocks — the audit measured blindness at ~15/16
    of arbitrary offsets. ORB keypoints are anchored to image content, not to
    a sampling grid, so matching is alignment-free (and tolerant of the mild
    pixel noise a JPEG re-save introduces).

    Three guards separate forgery from the structures genuine documents
    legitimately repeat:

    1. Repetition filter — a keypoint whose descriptor has more than
       ``max_neighbor_matches`` near-duplicates ANYWHERE in the image belongs
       to repeating texture (guilloche, QR modules, tiled emblems) and is
       excluded entirely. Measured before any spatial filtering so periodic
       patterns cannot hide behind a minimum-shift cut.
    2. Shift-vector clustering with a dominance test — a duplicated region
       moves by ONE offset, so forgery concentrates in a single displacement
       bin (``min_matches`` required) that clearly dominates the runner-up
       (``dominance_ratio``). Periodic structure that survives the repetition
       filter (ORB's keypoint budget can sample a sparse subset of a texture)
       spreads across many comparable bins and fails dominance.
    3. Region-span band — the dominant cluster's source keypoints must span
       a REGION: at least ``min_span_px`` on both axes (rejecting thin repeated
       text strips and tiny QR motifs) and at most ``max_span_px`` on both
       (rejecting periodic print structure, whose same-offset matches scatter
       card-wide instead of clustering into one pasted region). A genuine
       photo/field duplication is compact and 2-D; a guilloche or a monospace
       number's repeats are either 1-D or card-spanning.
       Deliberate trade-off: single-line text duplication is out of scope here
       (that is the text-splice attack surface).

    v4 (ADR-034): SIFT fallback. ORB's tight Hamming threshold (10) misses
    some duplicated photo regions where JPEG noise pushes descriptor distances
    past the cut. SIFT's 128-dim float descriptors with Lowe's ratio test are
    more tolerant. Runs only when ORB returns ``detected=False``, with a
    higher min_matches bar (12 vs 8) and tighter max_aspect (3.0 vs 4.0)
    to maintain 0% FPR.

    v5: three guards against same-shift text coincidences, found when the
    forge corpus was first rendered with real Windows fonts (4/60 genuine
    documents flagged: repeated digits at the Aadhaar group pitch, a label
    prefix shared with the next line, a name glyph coinciding with the photo
    frame's corners). Span is measured on the cluster's main spatial group,
    the SIFT stage uses a wider span floor, and every accepted cluster must
    pass a weak patch-NCC floor. Measured: 0/60 genuine, recall unchanged.
    """

    SIFT_RATIO_THRESH: float = 0.65
    SIFT_PROMISCUITY_L2: float = 200.0
    SIFT_MIN_MATCHES: int = 12
    SIFT_MAX_ASPECT: float = 3.0
    SIFT_MERGE_RADIUS: int = 1
    SIFT_MERGE_MIN: int = 25
    SIFT_PATCH_NCC_MIN: float = 0.78
    SIFT_PATCH_DOM_MIN: int = 17
    SIFT_PATCH_RATIO_MIN: float = 1.15
    # v5 guards (see class docstring). SIFT's multi-scale keypoints scatter
    # ~8 px wider than ORB's around the same glyphs: a repeated two-digit block
    # of a monospace ID number measured 28x23 px under ORB (rejected by
    # min_span_px) but 33x37 under SIFT. Glyph height at card scale is <= 37;
    # the smallest real SIFT-path region cluster measured 49.
    SIFT_MIN_SPAN_PX: float = 40.0
    CORE_LINK_PX: float = 96.0
    CORE_FRACTION: float = 0.75
    PATCH_NCC_FLOOR: float = 0.40

    def __init__(
        self,
        min_matches: int = 8,
        match_max_hamming: int = 10,
        min_shift_px: int = 48,
        max_neighbor_matches: int = 3,
        shift_bin_px: int = 8,
        min_span_px: float = 32.0,
        max_span_px: float = 340.0,
        max_aspect: float = 4.0,
        dominance_ratio: float = 2.0,
        max_keypoints: int = 5000,
        fast_threshold: int = 5,
    ) -> None:
        self.min_matches = min_matches
        self.match_max_hamming = match_max_hamming
        self.min_shift_px = min_shift_px
        self.max_neighbor_matches = max_neighbor_matches
        self.shift_bin_px = shift_bin_px
        self.min_span_px = min_span_px
        self.max_span_px = max_span_px
        self.max_aspect = max_aspect
        self.dominance_ratio = dominance_ratio
        self.max_keypoints = max_keypoints
        self.fast_threshold = fast_threshold

    def detect(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        result = self._orb_stage(gray)
        if not result["detected"]:
            result = self._sift_stage(gray)
        return result

    @staticmethod
    def _pair_shift(pts: np.ndarray, pairs: List[tuple]) -> tuple:
        """Median pixel displacement of the pairs. The bin centre can be off by
        up to shift_bin_px/2, which zeroes patch correlation on fine texture."""
        d = np.median(np.array([pts[j] - pts[i] for i, j, _ in pairs]), axis=0)
        return int(round(float(d[0]))), int(round(float(d[1])))

    def _patch_ncc(
        self, gray: np.ndarray, src_pts: np.ndarray, shift: tuple
    ) -> float:
        """Pearson correlation between the source region and its shifted copy."""
        h, w = gray.shape[:2]
        pad = 10
        x0, y0 = src_pts.min(axis=0).astype(int)
        x1, y1 = src_pts.max(axis=0).astype(int)
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        sx, sy = int(shift[0]), int(shift[1])
        tx0, ty0 = x0 + sx, y0 + sy
        tx1, ty1 = x1 + sx, y1 + sy
        if tx0 < 0:
            x0 -= tx0; tx0 = 0
        if ty0 < 0:
            y0 -= ty0; ty0 = 0
        if tx1 > w:
            x1 -= tx1 - w; tx1 = w
        if ty1 > h:
            y1 -= ty1 - h; ty1 = h
        if x1 - x0 < 20 or y1 - y0 < 20:
            return 0.0
        src = gray[y0:y1, x0:x1].astype(np.float64).ravel()
        tgt = gray[ty0:ty1, tx0:tx1].astype(np.float64).ravel()
        if src.shape != tgt.shape or len(src) < 100:
            return 0.0
        src_z = src - src.mean()
        tgt_z = tgt - tgt.mean()
        denom = np.sqrt((src_z ** 2).sum() * (tgt_z ** 2).sum())
        if denom < 1e-10:
            return 0.0
        return float((src_z * tgt_z).sum() / denom)

    def _core_points(self, src: np.ndarray) -> np.ndarray:
        """Points of the cluster's main spatial group, for span measurement.

        Single-linkage at ``CORE_LINK_PX``; the largest group is used when it
        holds at least ``CORE_FRACTION`` of the cluster, else the raw cluster.
        Without this, one same-shift coincidence 180 px away stretched a
        37x24 px repeated-digit block into a 105x198 "region". Measured on the
        forge corpus: real photo-region clusters have internal gaps up to
        ~70 px, strays sit >= 170 px away, so any radius in 80-160 separates
        them; 96 is two minimum shifts.
        """
        n = len(src)
        if n < 4:
            return src
        dist = np.hypot(src[:, 0, None] - src[None, :, 0], src[:, 1, None] - src[None, :, 1])
        adjacent = dist <= self.CORE_LINK_PX
        labels = np.full(n, -1)
        groups = 0
        for seed in range(n):
            if labels[seed] >= 0:
                continue
            labels[seed] = groups
            stack = [seed]
            while stack:
                u = stack.pop()
                for v in np.flatnonzero(adjacent[u] & (labels < 0)):
                    labels[v] = groups
                    stack.append(v)
            groups += 1
        main = src[labels == np.bincount(labels).argmax()]
        return main if len(main) >= self.CORE_FRACTION * n else src

    def _orb_stage(self, gray: np.ndarray) -> Dict[str, Any]:
        empty: Dict[str, Any] = {
            "detected": False,
            "matched_pairs": [],
            "confidence": 0.0,
            "dominant_shift": None,
        }

        orb = cv2.ORB_create(nfeatures=self.max_keypoints, fastThreshold=self.fast_threshold)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < self.min_matches * 2:
            return empty

        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        n = len(keypoints)

        k = min(self.max_neighbor_matches + 6, n)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(descriptors, descriptors, k=k)

        near_dups = np.zeros(n, dtype=int)
        for matches in knn:
            for m in matches:
                if m.queryIdx == m.trainIdx or m.distance > self.match_max_hamming:
                    continue
                d = pts[m.queryIdx] - pts[m.trainIdx]
                if float(np.hypot(d[0], d[1])) > 4.0:
                    near_dups[m.queryIdx] += 1
        keep = near_dups <= self.max_neighbor_matches

        seen = set()
        pairs: List[tuple] = []
        for matches in knn:
            for m in matches:
                i, j = m.queryIdx, m.trainIdx
                if i == j or not (keep[i] and keep[j]):
                    continue
                if m.distance > self.match_max_hamming:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                d = pts[j] - pts[i]
                if float(np.hypot(d[0], d[1])) < self.min_shift_px:
                    continue
                seen.add(key)
                pairs.append((key[0], key[1], m.distance))
        if not pairs:
            return empty

        return self._cluster_and_decide(pts, pairs, self.min_matches, self.max_aspect, 256.0,
                                        gray=gray)

    def _sift_stage(self, gray: np.ndarray) -> Dict[str, Any]:
        """SIFT fallback: 128-dim float descriptors with ratio test."""
        empty: Dict[str, Any] = {
            "detected": False,
            "matched_pairs": [],
            "confidence": 0.0,
            "dominant_shift": None,
        }

        sift = cv2.SIFT_create(nfeatures=self.max_keypoints)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < self.SIFT_MIN_MATCHES * 2:
            return empty

        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        n = len(keypoints)

        k = min(self.max_neighbor_matches + 6, n)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn = matcher.knnMatch(descriptors, descriptors, k=k)

        # C2 fix: sentinel values so short knnMatch results are safe.
        q_idx = np.arange(n, dtype=np.int32)[:, None].repeat(k, axis=1)
        t_idx = q_idx.copy()
        dists = np.full((n, k), np.inf, dtype=np.float32)
        for r, matches in enumerate(knn):
            for c, m in enumerate(matches):
                q_idx[r, c] = m.queryIdx
                t_idx[r, c] = m.trainIdx
                dists[r, c] = m.distance

        not_self = q_idx != t_idx
        dxy = pts[t_idx] - pts[q_idx]
        far_spatial = np.hypot(dxy[..., 0], dxy[..., 1]) > 4.0
        near_dups = (not_self & (dists < self.SIFT_PROMISCUITY_L2) & far_spatial).sum(axis=1)
        keep = near_dups <= self.max_neighbor_matches

        # S1 fix: L2 self-matching always places self-match at column 0
        # (distance 0), so best non-self is column 1, second-best is column 2.
        if k < 3:
            return empty
        arange = np.arange(n)
        best_idx = t_idx[:, 1]
        best_dist = dists[:, 1]
        second_dist = dists[:, 2]

        ratio_ok = best_dist / np.maximum(second_dist, 1e-6) <= self.SIFT_RATIO_THRESH
        both_keep = keep[arange] & keep[best_idx]
        shift_vec = pts[best_idx] - pts
        shift_len = np.hypot(shift_vec[:, 0], shift_vec[:, 1])
        far_enough = shift_len >= self.min_shift_px
        valid = ratio_ok & both_keep & far_enough

        seen = set()
        pairs: List[tuple] = []
        for i in np.where(valid)[0]:
            j = int(best_idx[i])
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((key[0], key[1], float(best_dist[i])))
        if not pairs:
            return empty

        return self._cluster_and_decide(
            pts, pairs, self.SIFT_MIN_MATCHES, self.SIFT_MAX_ASPECT,
            self.SIFT_PROMISCUITY_L2,
            min_span=self.SIFT_MIN_SPAN_PX,
            merge_radius=self.SIFT_MERGE_RADIUS,
            merge_min=self.SIFT_MERGE_MIN,
            gray=gray,
            patch_ncc_min=self.SIFT_PATCH_NCC_MIN,
            patch_dom_min=self.SIFT_PATCH_DOM_MIN,
            patch_ratio_min=self.SIFT_PATCH_RATIO_MIN,
        )

    def _cluster_and_decide(
        self,
        pts: np.ndarray,
        pairs: List[tuple],
        min_matches: int,
        max_aspect: float,
        dist_scale: float,
        merge_radius: int = 0,
        merge_min: int = 25,
        gray: np.ndarray | None = None,
        patch_ncc_min: float = 0.0,
        patch_dom_min: int = 0,
        patch_ratio_min: float = 0.0,
        min_span: float | None = None,
    ) -> Dict[str, Any]:
        bins: List[tuple] = []
        oriented: List[tuple] = []
        for i, j, dist in pairs:
            dx, dy = pts[j] - pts[i]
            if dx < 0 or (dx == 0 and dy < 0):
                i, j = j, i
                dx, dy = -dx, -dy
            bins.append((round(dx / self.shift_bin_px), round(dy / self.shift_bin_px)))
            oriented.append((i, j, dist))
        tally = Counter(bins)
        top = tally.most_common(2)
        dom_bin, dom_count = top[0]
        runner_up = top[1][1] if len(top) > 1 else 0
        dominant = dom_count >= max(min_matches, self.dominance_ratio * runner_up)
        dom_pairs = [p for p, b in zip(oriented, bins) if b == dom_bin]
        shift_px = self._pair_shift(pts, dom_pairs)

        if min_span is None:
            min_span = self.min_span_px

        def span_ok_for(points: np.ndarray) -> bool:
            ext = points.max(axis=0) - points.min(axis=0) if len(points) > 1 else np.zeros(2)
            lo, hi = float(min(ext)), float(max(ext))
            return min_span <= lo and hi <= self.max_span_px and hi / max(lo, 1.0) <= max_aspect

        src = self._core_points(np.array([pts[i] for i, _, _ in dom_pairs]))
        span_ok = span_ok_for(src)

        detected = dominant and span_ok
        verified = False

        if not detected and merge_radius > 0:
            neighborhood = set()
            for db in range(-merge_radius, merge_radius + 1):
                for dd in range(-merge_radius, merge_radius + 1):
                    neighborhood.add((dom_bin[0] + db, dom_bin[1] + dd))
            merged_count = sum(tally.get(nb, 0) for nb in neighborhood)
            nbr_runner = max((c for b, c in tally.items()
                              if b not in neighborhood), default=0)
            nbr_dominant = merged_count >= max(merge_min,
                                               self.dominance_ratio * nbr_runner)
            if nbr_dominant:
                nbr_pairs = [p for p, b in zip(oriented, bins)
                             if b in neighborhood]
                nbr_src = self._core_points(np.array([pts[i] for i, _, _ in nbr_pairs]))
                if span_ok_for(nbr_src):
                    detected = True
                    dom_pairs = nbr_pairs
                    dom_count = merged_count
                    src = nbr_src
                    shift_px = self._pair_shift(pts, dom_pairs)

        if (not detected and gray is not None and patch_ncc_min > 0
                and span_ok
                and dom_count >= patch_dom_min
                and dom_count > patch_ratio_min * runner_up):
            ncc = self._patch_ncc(gray, src, shift_px)
            if ncc >= patch_ncc_min:
                detected = True
                verified = True

        # Every accepted cluster must show at least weak pixel agreement
        # between its box and the shifted box. Keypoint-only evidence can be a
        # composite of unrelated coincidences at one shift — a repeated name
        # glyph plus the two bottom corners of the photo frame measured 0.08,
        # a label prefix shared with the next text line 0.18 — while real
        # duplications measured >= 0.54 on the forge corpus.
        if (detected and not verified and gray is not None
                and self._patch_ncc(gray, src, shift_px) < self.PATCH_NCC_FLOOR):
            detected = False

        matched_pairs = [
            {
                "block1": [int(pts[i][0]), int(pts[i][1])],
                "block2": [int(pts[j][0]), int(pts[j][1])],
                "similarity": round(max(0.0, 1.0 - dist / dist_scale), 4),
            }
            for i, j, dist in dom_pairs[:20]
        ]

        confidence = min(1.0, dom_count / (min_matches * 2))

        return {
            "detected": detected,
            "matched_pairs": matched_pairs,
            "confidence": confidence if detected else min(confidence, 0.3),
            "dominant_shift": [
                int(dom_bin[0] * self.shift_bin_px),
                int(dom_bin[1] * self.shift_bin_px),
            ],
        }
