import os
import os.path as osp

import argparse
import pickle
import torch
import smplx

from cmd_parser import parse_config
from human_body_prior.tools.model_loader import load_vposer

from utils import JointMapper
import pyrender
import trimesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', nargs='+', type=str, required=True,
                        help='The pkl files that will be read')

    args, remaining = parser.parse_known_args()

    pkl_paths = args.pkl

    args = parse_config(remaining)
    dtype = torch.float32
    use_cuda = args.get('use_cuda', True)
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(args.get('model_folder'))
    model_params = dict(model_path=args.get('model_folder'),
                        #  joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    model = smplx.create(**model_params)
    model = model.to(device=device)

    batch_size = args.get('batch_size', 1)
    use_vposer = args.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    vposer_ckpt = args.get('vposer_ckpt', '')
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        if use_vposer:
            with torch.no_grad():
                pose_embedding[:] = torch.tensor(
                    data['body_pose'], device=device, dtype=dtype)

        est_params = {}
        for key, val in data.items():
            if key == 'body_pose' and use_vposer:
                est_params['body_pose'] = vposer.decode(
                    pose_embedding, output_type='aa').view(1, -1)
            else:
                est_params[key] = torch.tensor(val, dtype=dtype, device=device)

        model_output = model(**est_params)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')
        pyrender.Viewer(scene, use_raymond_lighting=True)
