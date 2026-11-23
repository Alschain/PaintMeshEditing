import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F

from util.losses import (
    VGGPerceptualLoss,
    area_preservation_loss,
    depth_map_loss,
    differential_laplacian_loss,
    edge_length_loss,
    gaussian_blur_hwc,
    image_l2_loss,
    laplacian_loss,
    normal_map_loss,
    relative_normal_consistency_loss,
    silhouette_loss,
    ssim_loss,
    vertex_displacement_loss,
)
from util.mesh_util import (
    compute_edge_length,
    compute_laplacian_uniform,
    compute_normals,
    compute_triangle_area,
)
from util.vis_util import vis_depth_map, vis_normal_map
from util.obj_io import save_obj
from core.adaptive_loss import AdaptiveLossBalancer


DEFAULT_LOSS_WEIGHTS = {
    # 2D image & multi-modal guidance terms
    'img_loss': 100.0,
    'img_l2_loss': 0.0,
    'ssim_loss': 0.0,
    'vgg_loss': 0.0,
    'normal_loss': 0.1,
    'depth_loss': 10.0,
    'silhouette_loss': 50.0,
    # 3D mesh geometry regularization terms
    'nc_loss': 10.0,
    'edge_loss': 100.0,
    'area_loss': 100000.0,
    'lap_loss': 0.0,
    'diff_lap_loss': 0.0,
    'disp_loss': 0.0,
}


class ViewConstraint:
    """Single camera viewpoint constraint for multi-view simultaneous PME optimization."""

    def __init__(
        self,
        intrinsic,
        extrinsic,
        label_image=None,
        mask_image=None,
        normal_image=None,
        depth_image=None,
        silhouette_image=None,
        render_size=None,
        weight=1.0,
        name='view',
    ):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.label_image = label_image
        self.mask_image = mask_image
        self.normal_image = normal_image
        self.depth_image = depth_image
        self.silhouette_image = silhouette_image
        self.render_size = render_size
        self.weight = float(weight)
        self.name = str(name)

    @classmethod
    def from_item(cls, item, idx=0):
        if isinstance(item, cls):
            return item
        if isinstance(item, dict):
            return cls(
                intrinsic=item['intrinsic'],
                extrinsic=item['extrinsic'],
                label_image=item.get('label_image'),
                mask_image=item.get('mask_image'),
                normal_image=item.get('normal_image'),
                depth_image=item.get('depth_image'),
                silhouette_image=item.get('silhouette_image'),
                render_size=item.get('render_size'),
                weight=item.get('weight', 1.0),
                name=item.get('name', f'view_{idx}'),
            )
        raise TypeError(f"Unsupported view constraint type: {type(item)}")


def _prep_tensor(t, device, ensure_channel=False):
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        t = torch.FloatTensor(t)
    t = t.to(device).float()
    if ensure_channel and t.ndim == 2:
        t = t.unsqueeze(-1)
    return t


class PaintMeshEditing:
    def __init__(self, args=None):
        args = args or {}

        self.iteration = int(args.get('iteration', 2000))
        self.lr = float(args.get('lr', 3e-4))
        self.lr_decay = float(args.get('lr_decay', 0.0002))
        self.log_interval = int(args.get('log_interval', 10))
        # Keep backward-compatible fallback for legacy typo 'save_iterval'
        self.save_interval = int(args.get('save_interval', args.get('save_iterval', 20)))
        self.display_image = bool(args.get('display_image', True))
        self.save_mesh_interval = int(args.get('save_mesh_interval', 100))

        self.SCALE = args.get('SCALE', 1)
        self.USE_BLUR = bool(args.get('USE_BLUR', False))
        self.BLUR_KERNEL_SIZE = int(args.get('BLUR_KERNEL_SIZE', 9))

        # Loss weights
        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        if 'loss_weights' in args and isinstance(args['loss_weights'], dict):
            for k, v in args['loss_weights'].items():
                self.loss_weights[k] = float(v)

        # Automatic loss adaptation settings
        self.adaptive_mode = str(args.get('adaptive_mode', 'fixed')).lower()
        self.adapt_interval = int(args.get('adapt_interval', 10))
        self.adapt_ema = float(args.get('adapt_ema', 0.9))
        self.adapt_min_scale = float(args.get('adapt_min_scale', 0.1))
        self.adapt_max_scale = float(args.get('adapt_max_scale', 10.0))

        self._vgg_loss_fn = None

    def _get_vgg_loss(self, device):
        if self._vgg_loss_fn is None:
            self._vgg_loss_fn = VGGPerceptualLoss(resize=True).to(device)
        else:
            self._vgg_loss_fn = self._vgg_loss_fn.to(device)
        return self._vgg_loss_fn

    def _current_blur_kernel(self, iteration):
        """Coarse-to-fine blur kernel schedule: shrinks to 1 in the final third."""
        if not self.USE_BLUR or self.BLUR_KERNEL_SIZE <= 1:
            return 1
        progress = iteration / max(1, self.iteration - 1)
        if progress >= 0.7:
            return 1
        k = int(round(self.BLUR_KERNEL_SIZE * (1.0 - progress / 0.7)))
        if k <= 1:
            return 1
        return k if k % 2 == 1 else k + 1

    def _precompute_geometry_invariants(self, mesh):
        device = mesh.vs.device
        ori_vs = mesh.vs.detach().clone()
        ori_vertex_normals, ori_face_normals = compute_normals(ori_vs, mesh.f_v_idx)
        ori_link_nc = ori_vertex_normals[mesh.edges[:, 0]] - ori_vertex_normals[mesh.edges[:, 1]]
        if mesh.connected_faces is not None and mesh.connected_faces.numel() > 0:
            ori_link_face_nc = (
                ori_face_normals[mesh.connected_faces[:, 0]] - ori_face_normals[mesh.connected_faces[:, 1]]
            )
        else:
            ori_link_face_nc = torch.zeros((0, 3), device=device)

        ori_face_area = compute_triangle_area(ori_vs, mesh.f_v_idx)
        ori_edge_len = compute_edge_length(ori_vs, mesh.edges)
        laplacian = compute_laplacian_uniform(ori_vs, mesh.edges)
        ori_delta = torch.sparse.mm(laplacian, ori_vs).detach()

        return {
            'ori_vs': ori_vs,
            'ori_link_nc': ori_link_nc,
            'ori_link_face_nc': ori_link_face_nc,
            'ori_face_area': ori_face_area,
            'ori_edge_len': ori_edge_len,
            'laplacian': laplacian,
            'ori_delta': ori_delta,
        }

    def _compute_view_2d_losses(
        self,
        mesh,
        renderer,
        it,
        label_image,
        mask_image,
        normal_image=None,
        depth_image=None,
        silhouette_image=None,
        vgg_fn=None,
    ):
        losses = {}
        need_normal = normal_image is not None and self.loss_weights.get('normal_loss', 0.0) > 0.0
        render_result = renderer(mesh, render_normal=need_normal)

        # 1. Color / Label guidance losses
        if label_image is not None and render_result['color'] is not None:
            pred_color = render_result['color'][0]
            blur_k = self._current_blur_kernel(it)
            if blur_k > 1:
                pred_cmp = gaussian_blur_hwc(pred_color, blur_k)
                tgt_cmp = gaussian_blur_hwc(label_image, blur_k)
            else:
                pred_cmp = pred_color
                tgt_cmp = label_image

            if mask_image is not None:
                image_masked = pred_cmp * mask_image
                target_masked = tgt_cmp * mask_image if blur_k > 1 else tgt_cmp
            else:
                image_masked = pred_cmp
                target_masked = tgt_cmp

            if self.loss_weights.get('img_loss', 0.0) > 0.0:
                losses['img_loss'] = F.l1_loss(image_masked, target_masked)

            if self.loss_weights.get('img_l2_loss', 0.0) > 0.0:
                losses['img_l2_loss'] = image_l2_loss(pred_cmp, tgt_cmp, mask_image)

            if self.loss_weights.get('ssim_loss', 0.0) > 0.0:
                losses['ssim_loss'] = ssim_loss(pred_cmp, tgt_cmp, mask_image)

            if vgg_fn is not None and self.loss_weights.get('vgg_loss', 0.0) > 0.0:
                losses['vgg_loss'] = vgg_fn(image_masked, target_masked)

        # 2. Normal map guidance
        if need_normal:
            losses['normal_loss'] = normal_map_loss(
                render_result['normal'][0], normal_image, mask=mask_image
            )

        # 3. Depth map guidance
        if depth_image is not None and self.loss_weights.get('depth_loss', 0.0) > 0.0:
            losses['depth_loss'] = depth_map_loss(
                render_result['depth'][0], depth_image, mask=mask_image
            )

        # 4. Silhouette guidance
        if silhouette_image is not None and self.loss_weights.get('silhouette_loss', 0.0) > 0.0:
            sil_pred = render_result.get('silhouette', render_result['mask'])[0]
            losses['silhouette_loss'] = silhouette_loss(sil_pred, silhouette_image)

        return losses

    def _compute_geometry_losses(self, mesh, inv):
        losses = {}
        if self.loss_weights.get('lap_loss', 0.0) > 0.0:
            losses['lap_loss'] = laplacian_loss(inv['laplacian'], mesh.vs)

        if self.loss_weights.get('diff_lap_loss', 0.0) > 0.0:
            losses['diff_lap_loss'] = differential_laplacian_loss(inv['laplacian'], mesh.vs, inv['ori_delta'])

        if self.loss_weights.get('nc_loss', 0.0) > 0.0:
            vertex_normals, face_normals = compute_normals(mesh.vs, mesh.f_v_idx)
            losses['nc_loss'] = relative_normal_consistency_loss(
                vertex_normals,
                face_normals,
                mesh.edges,
                mesh.connected_faces,
                inv['ori_link_nc'],
                inv['ori_link_face_nc'],
            )

        if self.loss_weights.get('edge_loss', 0.0) > 0.0:
            edges_len = compute_edge_length(mesh.vs, mesh.edges)
            losses['edge_loss'] = edge_length_loss(edges_len, inv['ori_edge_len'])

        if self.loss_weights.get('area_loss', 0.0) > 0.0:
            face_area = compute_triangle_area(mesh.vs, mesh.f_v_idx)
            losses['area_loss'] = area_preservation_loss(face_area, inv['ori_face_area'])

        if self.loss_weights.get('disp_loss', 0.0) > 0.0:
            losses['disp_loss'] = vertex_displacement_loss(mesh.vs, inv['ori_vs'])

        return losses

    def _render_view_preview(self, mesh, renderer, label_image, normal_image, depth_image):
        with torch.no_grad():
            vis_result = renderer(mesh, render_normal=(normal_image is not None))
            panels = []
            if vis_result['color'] is not None:
                panels.append(vis_result['color'][0].detach().cpu())
            if normal_image is not None and 'normal' in vis_result:
                normal_map_show = vis_normal_map(vis_result['normal'][0].detach().cpu()) / 255.0
                panels.append(normal_map_show)
            if depth_image is not None and 'depth' in vis_result:
                depth_map_show = vis_depth_map(vis_result['depth'][0].detach().cpu()) / 255.0
                panels.append(depth_map_show)
            if label_image is not None:
                panels.append(label_image.detach().cpu())
            if not panels:
                return None
            result_image = torch.cat(panels, dim=1)
            return np.clip(result_image.numpy() * 255.0, 0, 255).astype(np.uint8)

    def optimize(
        self,
        mesh,
        label_image,
        mask_image,
        renderer,
        normal_image=None,
        depth_image=None,
        silhouette_image=None,
        out_dir='./output',
        callback=None,
        stop_event=None,
        views=None,
    ):
        """Optimize mesh vertices from a single view or a list of multi-view constraints."""
        if views is not None and len(views) > 0:
            return self.optimize_multiview(
                mesh=mesh,
                views=views,
                renderer=renderer,
                out_dir=out_dir,
                callback=callback,
                stop_event=stop_event,
            )

        single_view = ViewConstraint(
            intrinsic=getattr(renderer, 'intrinsic', None),
            extrinsic=getattr(renderer, 'extrinsic', None),
            label_image=label_image,
            mask_image=mask_image,
            normal_image=normal_image,
            depth_image=depth_image,
            silhouette_image=silhouette_image,
            render_size=getattr(renderer, 'render_size', None),
            weight=1.0,
            name='view_0',
        )
        return self.optimize_multiview(
            mesh=mesh,
            views=[single_view],
            renderer=renderer,
            out_dir=out_dir,
            callback=callback,
            stop_event=stop_event,
            _single_view_keep_camera=True,
        )

    def optimize_multiview(
        self,
        mesh,
        views,
        renderer,
        out_dir='./output',
        callback=None,
        stop_event=None,
        _single_view_keep_camera=False,
    ):
        """Jointly optimize mesh vertices across multiple camera viewpoints simultaneously.

        Each item in `views` can be a `ViewConstraint` or a dict containing:
            intrinsic, extrinsic, label_image, mask_image, normal_image,
            depth_image, silhouette_image, render_size, weight, name.
        """
        if not views:
            raise ValueError("`views` must contain at least one ViewConstraint.")

        os.makedirs(out_dir, exist_ok=True)
        device = mesh.vs.device

        if not mesh.vs.requires_grad:
            mesh.vs.requires_grad_(True)

        parsed_views = []
        any_label = False
        for idx, v in enumerate(views):
            vc = ViewConstraint.from_item(v, idx=idx)
            vc.label_image = _prep_tensor(vc.label_image, device)
            vc.mask_image = _prep_tensor(vc.mask_image, device, ensure_channel=True)
            vc.normal_image = _prep_tensor(vc.normal_image, device)
            vc.depth_image = _prep_tensor(vc.depth_image, device, ensure_channel=True)
            vc.silhouette_image = _prep_tensor(vc.silhouette_image, device, ensure_channel=True)
            if vc.label_image is not None:
                any_label = True
            parsed_views.append(vc)

        inv = self._precompute_geometry_invariants(mesh)

        optimizer = torch.optim.Adam([mesh.vs], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: max(0.0, 10 ** (-x * self.lr_decay))
        )

        balancer = AdaptiveLossBalancer(
            base_weights=self.loss_weights,
            mode=self.adaptive_mode,
            adapt_interval=self.adapt_interval,
            ema_momentum=self.adapt_ema,
            min_scale=self.adapt_min_scale,
            max_scale=self.adapt_max_scale,
        )

        need_vgg = self.loss_weights.get('vgg_loss', 0.0) > 0.0 and any_label
        vgg_fn = self._get_vgg_loss(device) if need_vgg else None

        img_cnt = 0
        history = []
        completed_iters = 0

        for it in range(self.iteration):
            if stop_event is not None and stop_event.is_set():
                break

            # 1. Accumulate 2D losses across all camera viewpoints
            view_loss_accum = {}
            view_weight_accum = {}
            for vc in parsed_views:
                if not _single_view_keep_camera:
                    if vc.render_size is not None and hasattr(renderer, 'set_render_size'):
                        renderer.set_render_size(vc.render_size)
                    if vc.intrinsic is not None and hasattr(renderer, 'set_camera_intrinsic'):
                        renderer.set_camera_intrinsic(vc.intrinsic)
                    if vc.extrinsic is not None and hasattr(renderer, 'set_camera_extrinsic'):
                        renderer.set_camera_extrinsic(vc.extrinsic)

                v_losses = self._compute_view_2d_losses(
                    mesh=mesh,
                    renderer=renderer,
                    it=it,
                    label_image=vc.label_image,
                    mask_image=vc.mask_image,
                    normal_image=vc.normal_image,
                    depth_image=vc.depth_image,
                    silhouette_image=vc.silhouette_image,
                    vgg_fn=vgg_fn,
                )
                vw = max(vc.weight, 1e-6)
                for k, val in v_losses.items():
                    view_loss_accum[k] = view_loss_accum.get(k, 0.0) + val * vw
                    view_weight_accum[k] = view_weight_accum.get(k, 0.0) + vw

            losses = {k: view_loss_accum[k] / view_weight_accum[k] for k in view_loss_accum}

            # 2. Compute 3D geometry regularizers once per step
            geom_losses = self._compute_geometry_losses(mesh, inv)
            losses.update(geom_losses)

            # 3. Automatic loss weight adaptation
            cur_weights = balancer.update(it, losses, params=mesh.vs)

            all_loss = torch.zeros((), device=device)
            for key, value in losses.items():
                w = cur_weights.get(key, 0.0)
                if w > 0.0:
                    all_loss = all_loss + value * w

            optimizer.zero_grad()
            if all_loss.requires_grad:
                all_loss.backward()
                optimizer.step()
            scheduler.step()
            completed_iters = it + 1

            should_log = bool(self.log_interval and ((it + 1) % self.log_interval) == 0)
            should_save_img = bool(self.save_interval and (it % self.save_interval) == 0)
            should_preview = bool(self.display_image or should_save_img or (callback is not None and should_log))

            weighted_dict = {k: float(v.detach().item()) * cur_weights.get(k, 0.0) for k, v in losses.items()}
            total_loss_val = float(all_loss.detach().item())

            if should_log:
                prefix = f'iter \t{it} ({len(parsed_views)}v),' if len(parsed_views) > 1 else f'iter \t{it},'
                logline = prefix
                for key, w_val in weighted_dict.items():
                    logline += '\t {0}:{1:.6f},'.format(key, w_val)
                logline += '\t total:{0:.6f}'.format(total_loss_val)
                print(logline)
                history.append({
                    'iter': it + 1,
                    'total': total_loss_val,
                    'losses': weighted_dict,
                    'weights': dict(cur_weights),
                })

            preview_bgr = None
            if should_preview:
                strips = []
                max_preview_views = min(len(parsed_views), 4)
                for vc in parsed_views[:max_preview_views]:
                    if not _single_view_keep_camera:
                        if vc.render_size is not None and hasattr(renderer, 'set_render_size'):
                            renderer.set_render_size(vc.render_size)
                        if vc.intrinsic is not None and hasattr(renderer, 'set_camera_intrinsic'):
                            renderer.set_camera_intrinsic(vc.intrinsic)
                        if vc.extrinsic is not None and hasattr(renderer, 'set_camera_extrinsic'):
                            renderer.set_camera_extrinsic(vc.extrinsic)
                    s_img = self._render_view_preview(
                        mesh, renderer, vc.label_image, vc.normal_image, vc.depth_image
                    )
                    if s_img is not None:
                        strips.append(s_img)

                if strips:
                    if len(strips) == 1:
                        preview_bgr = strips[0]
                    else:
                        # Resize strips to uniform width and stack vertically for clear multi-view comparison
                        target_w = min(s.shape[1] for s in strips)
                        norm_strips = []
                        for s in strips:
                            if s.shape[1] != target_w:
                                new_h = max(1, int(round(s.shape[0] * target_w / s.shape[1])))
                                s = cv2.resize(s, (target_w, new_h), interpolation=cv2.INTER_AREA)
                            norm_strips.append(s)
                        preview_bgr = np.concatenate(norm_strips, axis=0)

                if preview_bgr is not None:
                    if self.display_image:
                        cv2.imshow('display', preview_bgr)
                        cv2.waitKey(1)
                    if should_save_img:
                        cv2.imwrite(os.path.join(out_dir, 'img_%06d.png' % img_cnt), preview_bgr)
                        img_cnt += 1

            if self.save_mesh_interval and ((it + 1) % self.save_mesh_interval) == 0:
                save_obj(os.path.join(out_dir, 'mesh', 'mesh_%06d.obj' % (it + 1)), mesh)

            if callback is not None and (should_log or it == 0 or it == self.iteration - 1):
                callback({
                    'iter': it + 1,
                    'total_iters': self.iteration,
                    'total_loss': total_loss_val,
                    'losses': weighted_dict,
                    'weights': dict(cur_weights),
                    'preview_bgr': preview_bgr,
                    'num_views': len(parsed_views),
                })

        return {
            'completed_iters': completed_iters,
            'history': history,
            'final_weights': balancer.get_weights(),
            'num_views': len(parsed_views),
        }
