import pyceres
import pycolmap
import numpy as np


def create_reconstruction(num_points=50, num_images=2, seed=3, noise=0):
    state = np.random.RandomState(seed)
    rec = pycolmap.Reconstruction()
    p3d = state.uniform(-1, 1, (num_points, 3)) + np.array([0, 0, 3])
    for p in p3d:
        rec.add_point3D(p, pycolmap.Track(), np.zeros(3))
    w, h = 640, 480
    cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=w,
        height=h,
        params=np.array([max(w, h) * 1.2, w / 2, h / 2]),
        camera_id=0,
    )
    rec.add_camera(cam)
    for i in range(num_images):
        im = pycolmap.Image(
            id=i,
            name=str(i),
            camera_id=cam.camera_id,
            cam_from_world=pycolmap.Rigid3d(
                pycolmap.Rotation3d(), state.uniform(-1, 1, 3)
            ),
        )
        im.registered = True
        p2d = cam.img_from_cam(
            im.cam_from_world * [p.xyz for p in rec.points3D.values()]
        )
        p2d_obs = np.array(p2d) + state.randn(len(p2d), 2) * noise
        im.points2D = pycolmap.ListPoint2D(
            [pycolmap.Point2D(p, id_) for p, id_ in zip(p2d_obs, rec.points3D)]
        )
        rec.add_image(im)
    return rec


rec_gt = create_reconstruction()

def define_problem(rec):
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        for p in im.points2D:
            cost = pycolmap.cost_functions.ReprojErrorCost(
                cam.model, im.cam_from_world, p.xy
            )
            prob.add_residual_block(
                cost, loss, [rec.points3D[p.point3D_id].xyz, cam.params]
            )
    for cam in rec.cameras.values():
        prob.set_parameter_block_constant(cam.params)
    return prob

def solve(prob):
    print(
        prob.num_parameter_blocks(),
        prob.num_parameters(),
        prob.num_residual_blocks(),
        prob.num_residuals(),
    )
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print(summary.BriefReport())


rec = create_reconstruction()
problem = define_problem(rec)
solve(problem)


rec = create_reconstruction()
for p in rec.points3D.values():
    p.xyz += np.random.RandomState(0).uniform(-0.5, 0.5, 3)
print(rec.points3D[1].xyz)
problem = define_problem(rec)
solve(problem)

def define_problem2(rec):
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        for p in im.points2D:
            cost = pycolmap.cost_functions.ReprojErrorCost(cam.model, p.xy)
            pose = im.cam_from_world
            params = [
                pose.rotation.quat,
                pose.translation,
                rec.points3D[p.point3D_id].xyz,
                cam.params,
            ]
            prob.add_residual_block(cost, loss, params)
        prob.set_manifold(
            im.cam_from_world.rotation.quat, pyceres.EigenQuaternionManifold()
        )
    for cam in rec.cameras.values():
        prob.set_parameter_block_constant(cam.params)
    for p in rec.points3D.values():
        prob.set_parameter_block_constant(p.xyz)
    return prob


rec = create_reconstruction()
for im in rec.images.values():
    im.cam_from_world.translation += np.random.randn(3) / 2
    im.cam_from_world.rotation *= pycolmap.Rotation3d(np.random.randn(3) / 5)
    im.cam_from_world.rotation.normalize()
rec_init = rec.__deepcopy__({})
init_from_gt = [
    rec.images[i].cam_from_world * rec_gt.images[i].cam_from_world.inverse()
    for i in rec.images
]
print([np.linalg.norm(t.translation) for t in init_from_gt])
print([np.rad2deg(t.rotation.angle()) for t in init_from_gt])
problem = define_problem2(rec)
solve(problem)
opt_from_gt = [
    rec.images[i].cam_from_world * rec_gt.images[i].cam_from_world.inverse()
    for i in rec.images
]
print([np.linalg.norm(t.translation) for t in opt_from_gt])
print([np.rad2deg(t.rotation.angle()) for t in opt_from_gt])


assert np.allclose(rec.cameras[0].params, rec_gt.cameras[0].params)
for i in rec.images:
    print(
        rec.images[i].cam_from_world.translation,
        rec_gt.images[i].cam_from_world.translation,
    )
    print(
        rec.images[i].cam_from_world.rotation.quat,
        rec_gt.images[i].cam_from_world.rotation.quat,
    )
rec.points3D[1].xyz, rec_gt.points3D[1].xyz